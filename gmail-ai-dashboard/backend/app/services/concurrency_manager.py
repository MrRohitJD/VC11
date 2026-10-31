"""Database-backed atomic concurrency control across dynos for background jobs, sync, RAG, and MCP."""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from threading import Event, Lock, Thread, current_thread
from typing import Callable, Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.models.chat_quota import ChatConcurrencyLock, ChatRequestLeaseRecord

logger = logging.getLogger(__name__)

LEASE_TTL_SECONDS = 180
HEARTBEAT_INTERVAL_SECONDS = 30
SessionFactory = Callable[[], Session]


def _ensure_concurrency_lock_row(db: Session, identity: str) -> None:
    """Ensure a row exists in chat_concurrency_locks for row-level serialization."""
    if db.get(ChatConcurrencyLock, identity) is not None:
        return
    try:
        with db.begin_nested():
            db.add(ChatConcurrencyLock(identity=identity[:80]))
            db.flush()
    except IntegrityError:
        pass


def _lock_concurrency_identity(db: Session, identity: str) -> None:
    _ensure_concurrency_lock_row(db, identity)
    row = db.scalar(
        select(ChatConcurrencyLock)
        .where(ChatConcurrencyLock.identity == identity[:80])
        .with_for_update()
    )
    if row is None:
        raise RuntimeError(f"Could not acquire concurrency lock for {identity}")


def _release_lease_record(session_factory: SessionFactory, lease_id: str, identity: str) -> None:
    try:
        with session_factory() as db:
            _lock_concurrency_identity(db, identity)
            db.execute(
                delete(ChatRequestLeaseRecord)
                .where(ChatRequestLeaseRecord.lease_id.in_([lease_id, f"{lease_id}:global"]))
            )
            db.commit()
    except Exception as exc:
        logger.warning("Could not delete concurrency lease %s: %s", lease_id, type(exc).__name__)


class OperationConcurrencyLease:
    """Active lease maintained with background heartbeat until released."""

    def __init__(self, lease_id: str, identity: str, session_factory: SessionFactory):
        self.lease_id = lease_id
        self.identity = identity
        self._session_factory = session_factory
        self._stop = Event()
        self._release_lock = Lock()
        self._released = False
        self._thread = Thread(
            target=self._heartbeat,
            name=f"lease-heartbeat-{identity[:16]}",
            daemon=True,
        )
        self._thread.start()

    def _heartbeat(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            try:
                now = datetime.now(timezone.utc)
                new_exp = now + timedelta(seconds=LEASE_TTL_SECONDS)
                with self._session_factory() as db:
                    _lock_concurrency_identity(db, self.identity)
                    changed = db.execute(
                        update(ChatRequestLeaseRecord)
                        .where(ChatRequestLeaseRecord.lease_id.in_([self.lease_id, f"{self.lease_id}:global"]))
                        .values(expires_at=new_exp)
                    ).rowcount
                    db.commit()
                if not changed:
                    return
            except Exception as exc:
                logger.warning("Lease heartbeat refresh failed: %s", type(exc).__name__)

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
            self._stop.set()

        if self._thread is not current_thread():
            self._thread.join(timeout=2)
        _release_lease_record(self._session_factory, self.lease_id, self.identity)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def acquire_operation_lease(
    db: Session,
    op_type: str,
    user_id: UUID | str,
    user_limit: int = 1,
    global_limit: int = 2,
    error_message: Optional[str] = None,
) -> OperationConcurrencyLease:
    """Acquire a concurrency lease for an expensive operation (e.g., sync, index, mcp).

    Enforces both per-user and global concurrency across all dynos.
    """
    user_identity = f"{op_type}:{user_id}"[:80]
    global_identity = f"{op_type}:global"[:80]
    now = datetime.now(timezone.utc)
    lease_id = secrets.token_hex(24)

    try:
        # Lock user identity
        _lock_concurrency_identity(db, user_identity)

        # Reap expired leases
        db.execute(
            delete(ChatRequestLeaseRecord).where(ChatRequestLeaseRecord.expires_at <= now)
        )

        user_active = db.scalar(
            select(func.count(ChatRequestLeaseRecord.lease_id)).where(
                ChatRequestLeaseRecord.identity == user_identity,
                ChatRequestLeaseRecord.expires_at > now,
            )
        ) or 0

        if user_active >= user_limit:
            msg = error_message or f"Another {op_type} job is currently running for your account. Please wait for it to finish."
            raise HTTPException(status_code=429, detail=msg)

        # Lock global identity
        _lock_concurrency_identity(db, global_identity)
        global_active = db.scalar(
            select(func.count(ChatRequestLeaseRecord.lease_id)).where(
                ChatRequestLeaseRecord.identity == global_identity,
                ChatRequestLeaseRecord.expires_at > now,
            )
        ) or 0

        if global_active >= global_limit:
            msg = error_message or f"System capacity for {op_type} is currently full across the application. Please try again shortly."
            raise HTTPException(status_code=429, detail=msg)

        # Create lease records
        exp = now + timedelta(seconds=LEASE_TTL_SECONDS)
        db.add(ChatRequestLeaseRecord(lease_id=lease_id, identity=user_identity, expires_at=exp))
        db.add(ChatRequestLeaseRecord(lease_id=f"{lease_id}:global", identity=global_identity, expires_at=exp))
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise

    session_factory = sessionmaker(bind=db.get_bind(), autoflush=False, expire_on_commit=False)
    return OperationConcurrencyLease(lease_id, user_identity, session_factory)

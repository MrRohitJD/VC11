"""Atomically reserve daily assistant quota and concurrent chat capacity."""

import hashlib
import hmac
import logging
import math
import secrets
from collections.abc import Callable
from datetime import datetime, time as datetime_time, timedelta, timezone
from threading import Event, Lock, Thread, current_thread

from fastapi import HTTPException, Request, Response
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.services.developer_settings import llm_key_context
from app.models.chat_quota import ChatConcurrencyLock, ChatQuota, ChatRequestLeaseRecord
from app.models.email import User

logger = logging.getLogger(__name__)

DEMO_DAILY_LIMIT = 10
USER_DAILY_LIMIT = 50
DEMO_CONCURRENT_LIMIT = 4
USER_CONCURRENT_LIMIT = 5
DEMO_VISITOR_COOKIE = "gmail_ai_demo_visitor"

# Leases are refreshed while a request runs and expire automatically if a
# worker exits without releasing its slot (for example, during a dyno restart).
CHAT_LEASE_TTL = timedelta(minutes=3)
CHAT_LEASE_HEARTBEAT_SECONDS = 30
SessionFactory = Callable[[], Session]


def _digest(value: str) -> str:
    return hmac.new(settings.session_secret.encode(), value.encode(), hashlib.sha256).hexdigest()


def _visitor_id(request: Request) -> str | None:
    raw_cookie = request.cookies.get(DEMO_VISITOR_COOKIE, "")
    visitor, separator, signature = raw_cookie.partition(".")
    if separator and visitor and len(visitor) <= 128 and hmac.compare_digest(signature, _digest(visitor)):
        return visitor
    return None


def set_demo_visitor_cookie(request: Request, response: Response) -> None:
    """Keep a signed visitor identifier across demo sessions and logouts."""
    visitor = _visitor_id(request) or secrets.token_urlsafe(32)
    response.set_cookie(
        DEMO_VISITOR_COOKIE, f"{visitor}.{_digest(visitor)}",
        httponly=True, secure=settings.secure_cookies, samesite="lax",
        max_age=365 * 86400,
    )


def _identity_and_limits(request: Request, user: User) -> tuple[str, int, int]:
    if not user.is_demo:
        daily_limit = settings.developer_daily_chat_limit if llm_key_context.get() else USER_DAILY_LIMIT
        return f"user:{user.id}", daily_limit, USER_CONCURRENT_LIMIT

    # Each demo browser gets its own quota. Shared network addresses must not
    # combine unrelated visitors' daily allowances.
    visitor = _visitor_id(request)
    if visitor is None:
        raise HTTPException(status_code=401, detail="Please enter the demo again")
    return f"demo:{_digest(visitor)}", DEMO_DAILY_LIMIT, DEMO_CONCURRENT_LIMIT


def _ensure_lock_row(db: Session, identity: str) -> None:
    """Create the per-identity mutex row, tolerating a concurrent first use."""
    if db.get(ChatConcurrencyLock, identity) is not None:
        return
    try:
        with db.begin_nested():
            db.add(ChatConcurrencyLock(identity=identity))
            db.flush()
    except IntegrityError:
        # Another request inserted the identity row while this request waited.
        pass


def _lock_identity(db: Session, identity: str) -> None:
    _ensure_lock_row(db, identity)
    lock_row = db.scalar(
        select(ChatConcurrencyLock)
        .where(ChatConcurrencyLock.identity == identity)
        .with_for_update()
    )
    if lock_row is None:
        raise RuntimeError("Could not acquire the assistant concurrency lock")


def _delete_lease(session_factory: SessionFactory, lease_id: str, identity: str) -> None:
    """Remove a lease in a separate session, serializing with reservations."""
    try:
        with session_factory() as db:
            _lock_identity(db, identity)
            db.execute(
                delete(ChatRequestLeaseRecord)
                .where(ChatRequestLeaseRecord.lease_id == lease_id)
            )
            db.commit()
    except Exception as exc:
        logger.warning("Could not release assistant request lease: %s", type(exc).__name__)


class ChatConcurrencyLease:
    """Keep a database lease alive until the assistant request completes."""

    def __init__(self, lease_id: str, identity: str, session_factory: SessionFactory):
        self.lease_id = lease_id
        self.identity = identity
        self._session_factory = session_factory
        self._stop = Event()
        self._release_lock = Lock()
        self._released = False
        self._heartbeat_thread = Thread(
            target=self._heartbeat,
            name="chat-quota-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat(self) -> None:
        while not self._stop.wait(CHAT_LEASE_HEARTBEAT_SECONDS):
            try:
                now = datetime.now(timezone.utc)
                with self._session_factory() as db:
                    _lock_identity(db, self.identity)
                    changed = db.execute(
                        update(ChatRequestLeaseRecord)
                        .where(ChatRequestLeaseRecord.lease_id == self.lease_id)
                        .values(expires_at=now + CHAT_LEASE_TTL)
                    ).rowcount
                    db.commit()
                if not changed:
                    return
            except Exception as exc:
                # The TTL provides crash recovery if the database is briefly
                # unavailable. A later heartbeat can renew the lease if present.
                logger.warning("Could not refresh assistant request lease: %s", type(exc).__name__)

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
            self._stop.set()

        if self._heartbeat_thread is not current_thread():
            self._heartbeat_thread.join(timeout=2)
        _delete_lease(self._session_factory, self.lease_id, self.identity)


def reserve_chat_request(db: Session, request: Request, user: User) -> ChatConcurrencyLease:
    """Reserve daily quota and an in-flight slot atomically across workers."""
    identity, daily_limit, concurrency_limit = _identity_and_limits(request, user)
    now = datetime.now(timezone.utc)
    day = now.date()
    lease_id = secrets.token_hex(24)

    try:
        # All requests for one visitor/account serialize on this row. This
        # makes the daily count and concurrency check atomic across dynos.
        _lock_identity(db, identity)

        # Reap abandoned requests left behind by a crashed worker.
        db.execute(
            delete(ChatRequestLeaseRecord)
            .where(ChatRequestLeaseRecord.expires_at <= now)
        )
        active_count = db.scalar(
            select(func.count(ChatRequestLeaseRecord.lease_id)).where(
                ChatRequestLeaseRecord.identity == identity,
                ChatRequestLeaseRecord.expires_at > now,
            )
        ) or 0
        if active_count >= concurrency_limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"You can have up to {concurrency_limit} assistant requests running "
                    "at a time. Wait for one to finish and try again."
                ),
            )

        quota = db.get(ChatQuota, (identity, day))
        if quota is None:
            quota = ChatQuota(identity=identity, day=day, count=0)
            db.add(quota)
            db.flush()
        if quota.count >= daily_limit:
            next_reset = datetime.combine(day + timedelta(days=1), datetime_time.min, tzinfo=timezone.utc)
            retry_after = max(1, math.ceil((next_reset - now).total_seconds()))
            actor = "demo visitor" if user.is_demo else "signed-in account"
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Daily assistant limit reached ({daily_limit} requests per {actor}). "
                    "Try again after the UTC daily reset."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        quota.count += 1
        db.add(ChatRequestLeaseRecord(
            lease_id=lease_id,
            identity=identity,
            expires_at=now + CHAT_LEASE_TTL,
        ))
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise

    session_factory = sessionmaker(bind=db.get_bind(), autoflush=False, expire_on_commit=False)
    try:
        return ChatConcurrencyLease(lease_id, identity, session_factory)
    except Exception:
        _delete_lease(session_factory, lease_id, identity)
        raise

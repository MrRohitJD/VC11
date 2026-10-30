"""RAG Auto-Indexing Daemon & User Sign-in Trigger Service.

Provides:
- Immediate, non-blocking auto-embedding of unindexed emails upon user login / session check.
- Periodic background daemon that sweeps across all user mailboxes every 30s.
- Tenant isolation enforced for all user-level background operations.
- Content-hash idempotency (skips already-embedded, unchanged emails).
- Automatic recovery and graceful error handling without impacting main request threads.
"""

from __future__ import annotations

import logging
from contextvars import copy_context
import threading
import time
from typing import Dict, Optional, Set
from uuid import UUID

from app.database.connection import SessionLocal, CentralSessionLocal
from app.services.developer_settings import decrypt, database_url_context
from app.services.rag_indexer import default_rag_service

logger = logging.getLogger(__name__)

# Daemon synchronization
_daemon_lock = threading.Lock()
_daemon_started = False
_daemon_wake_event = threading.Event()

# Guard against duplicate concurrent background workers for the same user
_active_user_workers: Set[str] = set()
_user_workers_lock = threading.Lock()


def trigger_immediate_auto_indexing() -> None:
    """Signal the background RAG auto-indexing daemon to wake up immediately."""
    _daemon_wake_event.set()


def ensure_user_emails_embedded(user_id: UUID, session_factory=None) -> None:
    """Non-blocking check on user login / sync.

    Spawns an asynchronous background worker to embed any of the user's emails
    that are not yet indexed in the PostgreSQL ``email_embeddings`` table.
    """
    from sqlalchemy import select
    from app.models.email import User

    with (session_factory or SessionLocal)() as check_db:
        if check_db.scalar(select(User.is_demo).where(User.id == user_id)):
            return
    user_id_str = str(user_id)

    with _user_workers_lock:
        if user_id_str in _active_user_workers:
            logger.debug("Auto-indexing worker already running for user %s; skipping new thread", user_id_str)
            return
        _active_user_workers.add(user_id_str)

    def _worker():
        try:
            from sqlalchemy import select
            from app.models.email import GmailAccount, User
            # Resolve personal database URL if configured
            personal_url = None
            try:
                with CentralSessionLocal() as central_db:
                    u = central_db.get(User, user_id)
                    personal_url = decrypt(u.developer_database_url) if u and u.developer_database_url else None
            except Exception:
                pass

            token = database_url_context.set(personal_url)
            try:
                factory = session_factory or SessionLocal
                with factory() as db:
                    total_indexed = 0
                    total_chunks = 0
                    idle_retries = 0
                    while True:
                        unindexed_ids = default_rag_service.get_unindexed_email_ids(db, user_id=user_id, limit=100)
                        if unindexed_ids:
                            idle_retries = 0
                            logger.info(
                                "User %s signed in / synced. Auto-embedding %d unindexed emails...",
                                user_id_str,
                                len(unindexed_ids),
                            )
                            res = default_rag_service.index_unindexed_emails(db, user_id=user_id, batch_size=100)
                            total_indexed += res.get("emails_indexed", 0)
                            total_chunks += res.get("chunks_created", 0)
                            continue

                        # Check if inbox sync is still actively pulling emails
                        account = db.scalar(select(GmailAccount).where(GmailAccount.user_id == user_id))
                        if account and account.sync_status in ("INITIAL", "BACKFILL") and idle_retries < 10:
                            idle_retries += 1
                            time.sleep(2)
                            continue
                        break

                    # Purge any emails that could not be embedded so none remain
                    purged = default_rag_service.purge_unembedded_emails(db, user_id=user_id)
                    logger.info(
                        "Auto-embedding completed for user %s: %d emails indexed, %d chunks stored, %d unembedded purged",
                        user_id_str,
                        total_indexed,
                        total_chunks,
                        purged,
                    )
            finally:
                database_url_context.reset(token)
        except Exception as exc:
            logger.exception("Error in auto-embedding worker for user %s: %s", user_id_str, exc)
        finally:
            with _user_workers_lock:
                _active_user_workers.discard(user_id_str)

    context = copy_context()
    worker_thread = threading.Thread(
        target=lambda: context.run(_worker),
        name=f"rag-auto-indexer-{user_id_str[:8]}",
        daemon=True,
    )
    worker_thread.start()


def _run_global_rag_indexing_cycle(batch_size: int = 100, session_factory=None) -> Dict[str, int]:
    """Execute one sweep across all mailboxes to embed unindexed emails."""
    factory = session_factory or SessionLocal
    db = factory()
    try:
        total_indexed = 0
        total_chunks = 0
        while True:
            unindexed_ids = default_rag_service.get_unindexed_email_ids(db, limit=batch_size)
            if not unindexed_ids:
                break
            logger.info("Global RAG daemon detected %d unindexed emails. Embedding batch...", len(unindexed_ids))
            results = default_rag_service.index_unindexed_emails(db, batch_size=batch_size)
            total_indexed += results.get("emails_indexed", 0)
            total_chunks += results.get("chunks_created", 0)

        # Purge any unembedded emails that failed embedding
        purged = default_rag_service.purge_unembedded_emails(db)
        if total_indexed or purged:
            logger.info(
                "Global RAG daemon cycle complete: %d emails indexed, %d chunks created, %d unembedded purged",
                total_indexed,
                total_chunks,
                purged,
            )
        return {"total_unindexed": 0, "emails_indexed": total_indexed, "chunks_created": total_chunks}
    finally:
        db.close()


def _global_rag_auto_indexer_loop(interval_seconds: float = 30.0) -> None:
    """Continuous background daemon loop sweeping for unindexed emails."""
    logger.info("Global background RAG auto-indexer daemon loop started.")
    time.sleep(2)
    while True:
        try:
            _daemon_wake_event.clear()
            _run_global_rag_indexing_cycle()
            from sqlalchemy import select
            from app.models.email import User
            with CentralSessionLocal() as central:
                personal_urls = [decrypt(value) for value in central.scalars(
                    select(User.developer_database_url).where(
                        User.developer_database_url.is_not(None))).all()]
            for url in personal_urls:
                token = database_url_context.set(url)
                try:
                    _run_global_rag_indexing_cycle()
                except Exception as exc:
                    logger.warning("Personal embedding sweep failed: %s", type(exc).__name__)
                finally:
                    database_url_context.reset(token)
        except Exception as exc:
            logger.exception("Error in background RAG auto-indexer daemon: %s", exc)

        # Check periodically (default: 30s) or wake immediately on trigger
        _daemon_wake_event.wait(timeout=interval_seconds)


def start_global_rag_auto_indexer_daemon(interval_seconds: float = 30.0) -> None:
    """Idempotently start the continuous background auto-indexer daemon."""
    global _daemon_started
    with _daemon_lock:
        if _daemon_started:
            return
        _daemon_started = True

    thread = threading.Thread(
        target=_global_rag_auto_indexer_loop,
        args=(interval_seconds,),
        name="global-rag-auto-indexer-daemon",
        daemon=True,
    )
    thread.start()
    logger.info("Started global background RAG auto-indexer daemon (interval: %.1fs).", interval_seconds)

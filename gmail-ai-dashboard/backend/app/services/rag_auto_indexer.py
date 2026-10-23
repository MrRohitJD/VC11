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
import threading
import time
from typing import Dict, Optional, Set
from uuid import UUID

from app.database.connection import SessionLocal
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
    that are not yet indexed in ChromaDB collection 'gmail_email_chunks_v2'.
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
            factory = session_factory or SessionLocal
            db = factory()
            try:
                unindexed_ids = default_rag_service.get_unindexed_email_ids(db, user_id=user_id)
                if not unindexed_ids:
                    logger.debug("No unindexed emails found for user %s", user_id_str)
                    return

                logger.info(
                    "User %s signed in / synced with %d unindexed emails. Starting automatic embedding...",
                    user_id_str,
                    len(unindexed_ids),
                )
                res = default_rag_service.index_unindexed_emails(db, user_id=user_id)
                logger.info(
                    "Completed auto-embedding for user %s: %d emails indexed, %d chunks stored",
                    user_id_str,
                    res.get("emails_indexed", 0),
                    res.get("chunks_created", 0),
                )
            finally:
                db.close()
        except Exception as exc:
            logger.exception("Error in auto-embedding worker for user %s: %s", user_id_str, exc)
        finally:
            with _user_workers_lock:
                _active_user_workers.discard(user_id_str)

    worker_thread = threading.Thread(
        target=_worker,
        name=f"rag-auto-indexer-{user_id_str[:8]}",
        daemon=True,
    )
    worker_thread.start()


def _run_global_rag_indexing_cycle(batch_size: int = 50, session_factory=None) -> Dict[str, int]:
    """Execute one sweep across all mailboxes to embed unindexed emails."""
    factory = session_factory or SessionLocal
    db = factory()
    try:
        unindexed_ids = default_rag_service.get_unindexed_email_ids(db, limit=batch_size)
        if not unindexed_ids:
            return {"total_unindexed": 0, "emails_indexed": 0, "chunks_created": 0}

        logger.info(
            "Periodic RAG daemon detected %d unindexed emails. Starting batch embedding...",
            len(unindexed_ids),
        )
        results = default_rag_service.index_unindexed_emails(db, batch_size=batch_size)
        logger.info(
            "Periodic RAG daemon finished batch: %d emails indexed, %d chunks created",
            results.get("emails_indexed", 0),
            results.get("chunks_created", 0),
        )
        return results
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

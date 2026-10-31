"""PostgreSQL RAG indexing service with large-mailbox background worker, progress tracking, and pause/resume."""

from __future__ import annotations

import logging
import threading
from contextvars import copy_context
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import exists, func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.connection import SessionLocal
from app.models.email import Email, GmailAccount, User
from app.models.rag import EmailChunk, EmailEmbedding
from app.rag.indexing import RAGIndexingPipeline
from app.rag.pgvector_store import PostgresVectorStoreManager
from app.services.ai_budget import calculate_cost, check_global_ai_budget, estimate_tokens, log_ai_usage
from app.services.concurrency_manager import acquire_operation_lease
from app.services.developer_settings import database_url_context

logger = logging.getLogger(__name__)

# In-memory progress tracking for large-mailbox indexing jobs: {user_id_str: state_dict}
_JOB_STATES: Dict[str, Dict[str, Any]] = {}
_JOB_LOCK = threading.Lock()
_ACTIVE_WORKER_THREADS: Dict[str, threading.Thread] = {}


class _PostgresCollectionFacade:
    """Small compatibility surface for callers that used ``collection.count``."""

    def __init__(self, manager: PostgresVectorStoreManager):
        self.manager = manager

    def count(self, user_id=None) -> int:
        return self.manager.count(user_id=user_id)


class RAGIndexerService:
    """Compatibility and large-scale indexing API backed entirely by ``email_embeddings``."""

    def __init__(self, collection_name=None, embedding_provider=None):
        self.collection_name = collection_name or "postgres_halfvec_3072"
        self.embedding_provider = embedding_provider
        self.vector_store = PostgresVectorStoreManager(embedding_function=embedding_provider)
        self.vector_store.collection_name = self.collection_name
        self.pipeline = RAGIndexingPipeline(vector_store_manager=self.vector_store)
        self.collection = _PostgresCollectionFacade(self.vector_store)

    def count_user_emails(self, db: Session, user_id: UUID | str) -> tuple[int, int, int]:
        """Return (total_emails, indexed_emails, unindexed_emails) for a user."""
        user_uuid = UUID(str(user_id))
        total = db.scalar(
            select(func.count(Email.id))
            .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(GmailAccount.user_id == user_uuid)
        ) or 0
        indexed = (select(1).select_from(EmailChunk)
                   .join(EmailEmbedding, EmailEmbedding.chunk_id == EmailChunk.id)
                   .where(EmailChunk.email_id == Email.id,
                          EmailEmbedding.status == "INDEXED",
                          EmailEmbedding.embedding.is_not(None),
                          EmailEmbedding.embedding_dimensions == settings.embedding_dimensions))
        unindexed = db.scalar(
            select(func.count(Email.id))
            .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(GmailAccount.user_id == user_uuid, ~exists(indexed))
        ) or 0
        return total, max(0, total - unindexed), unindexed

    def get_job_state(self, user_id: UUID | str, db: Optional[Session] = None) -> Dict[str, Any]:
        """Return real-time or cached progress tracking for user's indexing job."""
        user_id_str = str(user_id)
        with _JOB_LOCK:
            state = _JOB_STATES.get(user_id_str)
            if state is not None:
                return dict(state)

        # Build initial/idle state from DB counts if available
        if db is not None:
            total, indexed, unindexed = self.count_user_emails(db, user_id)
            initial = {
                "job_id": None,
                "user_id": user_id_str,
                "status": "completed" if unindexed == 0 and total > 0 else "idle",
                "total_emails_discovered": total,
                "emails_processed": indexed,
                "emails_pending": unindexed,
                "emails_failed": 0,
                "chunks_generated": indexed * 2,  # approx
                "embeddings_generated": indexed * 2,
                "current_batch": 0,
                "last_checkpoint": None,
                "estimated_cost_usd": 0.0,
                "error": None,
                "pause_requested": False,
            }
            with _JOB_LOCK:
                _JOB_STATES[user_id_str] = initial
            return dict(initial)

        return {
            "job_id": None,
            "user_id": user_id_str,
            "status": "idle",
            "total_emails_discovered": 0,
            "emails_processed": 0,
            "emails_pending": 0,
            "emails_failed": 0,
            "chunks_generated": 0,
            "embeddings_generated": 0,
            "current_batch": 0,
            "last_checkpoint": None,
            "estimated_cost_usd": 0.0,
            "error": None,
            "pause_requested": False,
        }

    def index_email(self, db: Session, email_id: UUID, user_id: Optional[UUID] = None, force: bool = False) -> int:
        return self.pipeline.index_email(db, email_id, user_id=user_id, force=force)

    def delete_email_from_rag(self, db: Session, email_id: UUID) -> None:
        self.pipeline.delete_email(db, email_id)

    def get_unindexed_email_ids(self, db: Session, user_id: Optional[UUID] = None, limit: Optional[int] = None) -> List[UUID]:
        indexed = (select(1).select_from(EmailChunk)
                   .join(EmailEmbedding, EmailEmbedding.chunk_id == EmailChunk.id)
                   .where(EmailChunk.email_id == Email.id,
                          EmailEmbedding.status == "INDEXED",
                          EmailEmbedding.embedding.is_not(None),
                          EmailEmbedding.embedding_dimensions == settings.embedding_dimensions))
        stmt = (select(Email.id).join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
                .where(~exists(indexed))
                .order_by(Email.received_at.desc().nullslast()))
        if user_id:
            stmt = stmt.where(GmailAccount.user_id == UUID(str(user_id)))
        elif database_url_context.get() is None:
            stmt = stmt.join(User, User.id == GmailAccount.user_id).where(
                User.developer_database_url.is_(None))
        if limit:
            stmt = stmt.limit(limit)
        return list(db.scalars(stmt).all())

    def index_emails_bulk(
        self,
        db: Session,
        email_ids: List[UUID | str],
        user_id: Optional[UUID] = None,
        force: bool = False,
        batch_size: int = 50,
    ) -> Dict[str, int]:
        return self.pipeline.index_emails_bulk(
            db, email_ids, user_id=user_id, force=force, batch_size=batch_size
        )

    def index_unindexed_emails(self, db: Session, user_id: Optional[UUID] = None,
                               batch_size: Optional[int] = None) -> Dict[str, int]:
        ids = self.get_unindexed_email_ids(db, user_id=user_id, limit=batch_size)
        if not ids:
            if not batch_size:
                self.purge_unembedded_emails(db, user_id=user_id)
            return {"total_unindexed": 0, "emails_indexed": 0, "chunks_created": 0}

        bulk_res = self.pipeline.index_emails_bulk(
            db,
            ids,
            user_id=user_id,
            batch_size=min(50, batch_size) if batch_size else 50,
        )
        if not batch_size or len(ids) < batch_size:
            self.purge_unembedded_emails(db, user_id=user_id)
        return {
            "total_unindexed": len(ids),
            "emails_indexed": bulk_res["emails_indexed"],
            "chunks_created": bulk_res["chunks_created"],
        }

    def purge_unembedded_emails(self, db: Session, user_id: Optional[UUID] = None) -> int:
        return self.pipeline.purge_unembedded_emails(db, user_id=user_id)

    def pause_indexing(self, user_id: UUID | str) -> Dict[str, Any]:
        """Signal the active indexing worker to pause after finishing its current batch."""
        user_id_str = str(user_id)
        with _JOB_LOCK:
            state = _JOB_STATES.get(user_id_str)
            if state is None:
                state = {
                    "job_id": None,
                    "user_id": user_id_str,
                    "status": "paused",
                    "total_emails_discovered": 0,
                    "emails_processed": 0,
                    "emails_pending": 0,
                    "emails_failed": 0,
                    "chunks_generated": 0,
                    "embeddings_generated": 0,
                    "current_batch": 0,
                    "last_checkpoint": None,
                    "estimated_cost_usd": 0.0,
                    "error": None,
                    "pause_requested": True,
                }
                _JOB_STATES[user_id_str] = state
            else:
                state["pause_requested"] = True
                if state.get("status") in ("idle", "running"):
                    state["status"] = "paused"
            return dict(state)

    def resume_indexing(self, user_id: UUID | str, session_factory=None) -> Dict[str, Any]:
        """Resume a paused indexing job."""
        user_id_str = str(user_id)
        with _JOB_LOCK:
            state = _JOB_STATES.get(user_id_str)
            if state:
                state["pause_requested"] = False
        return self.start_indexing_job(UUID(user_id_str), session_factory=session_factory, force=False)

    def retry_indexing(self, user_id: UUID | str, session_factory=None) -> Dict[str, Any]:
        """Retry indexing unindexed or failed emails."""
        return self.resume_indexing(user_id, session_factory=session_factory)

    def start_indexing_job(
        self,
        user_id: UUID,
        session_factory=None,
        batch_size: int = 100,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Start or continue a background indexing job across all user emails with batch continuation."""
        user_id_str = str(user_id)
        factory = session_factory or SessionLocal

        with factory() as db:
            total, indexed, unindexed = self.count_user_emails(db, user_id)

        job_id = str(uuid4())
        with _JOB_LOCK:
            existing = _JOB_STATES.get(user_id_str)
            if existing and existing.get("status") == "running":
                # Worker thread already active
                existing["pause_requested"] = False
                return dict(existing)

            state = {
                "job_id": job_id,
                "user_id": user_id_str,
                "status": "running",
                "total_emails_discovered": total,
                "emails_processed": 0 if force else indexed,
                "emails_pending": total if force else unindexed,
                "emails_failed": 0,
                "chunks_generated": 0 if force else (indexed * 2),
                "embeddings_generated": 0 if force else (indexed * 2),
                "current_batch": 0,
                "last_checkpoint": datetime.now(timezone.utc).isoformat(),
                "estimated_cost_usd": 0.0,
                "error": None,
                "pause_requested": False,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
            }
            _JOB_STATES[user_id_str] = state

        def _worker():
            self._worker_loop(user_id, job_id, session_factory=factory, batch_size=batch_size, force=force)

        ctx = copy_context()
        th = threading.Thread(
            target=lambda: ctx.run(_worker),
            name=f"rag-index-worker-{user_id_str[:8]}",
            daemon=True,
        )
        with _JOB_LOCK:
            _ACTIVE_WORKER_THREADS[user_id_str] = th
        th.start()
        return dict(state)

    def _worker_loop(
        self,
        user_id: UUID,
        job_id: str,
        session_factory=None,
        batch_size: int = 100,
        force: bool = False,
    ) -> None:
        """Background worker loop: batches emails, enforces leases, updates checkpoints, handles pause/resume."""
        user_id_str = str(user_id)
        factory = session_factory or SessionLocal

        with _JOB_LOCK:
            existing_st = _JOB_STATES.get(user_id_str)
            was_paused = existing_st.get("pause_requested", False) if existing_st else False
            if user_id_str not in _JOB_STATES or _JOB_STATES[user_id_str].get("job_id") != job_id:
                _JOB_STATES[user_id_str] = {
                    "job_id": job_id,
                    "user_id": user_id_str,
                    "status": "paused" if was_paused else "running",
                    "total_emails_discovered": existing_st.get("total_emails_discovered", 0) if existing_st else 0,
                    "emails_processed": existing_st.get("emails_processed", 0) if existing_st else 0,
                    "emails_pending": existing_st.get("emails_pending", 0) if existing_st else 0,
                    "emails_failed": existing_st.get("emails_failed", 0) if existing_st else 0,
                    "chunks_generated": existing_st.get("chunks_generated", 0) if existing_st else 0,
                    "embeddings_generated": existing_st.get("embeddings_generated", 0) if existing_st else 0,
                    "current_batch": existing_st.get("current_batch", 0) if existing_st else 0,
                    "last_checkpoint": datetime.now(timezone.utc).isoformat(),
                    "estimated_cost_usd": existing_st.get("estimated_cost_usd", 0.0) if existing_st else 0.0,
                    "error": None,
                    "pause_requested": was_paused,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "completed_at": None,
                }

        try:
            with factory() as init_db:
                total, indexed, unindexed = self.count_user_emails(init_db, user_id)
                with _JOB_LOCK:
                    st = _JOB_STATES[user_id_str]
                    if st.get("total_emails_discovered", 0) == 0:
                        st["total_emails_discovered"] = total
                        st["emails_processed"] = 0 if force else indexed
                        st["emails_pending"] = total if force else unindexed
        except Exception:
            pass

        # 1. Acquire cross-dyno concurrency lease
        try:
            with factory() as lease_db:
                lease = acquire_operation_lease(
                    lease_db, "rag_index", user_id, user_limit=1, global_limit=2
                )
        except HTTPException as exc:
            with _JOB_LOCK:
                if user_id_str in _JOB_STATES:
                    _JOB_STATES[user_id_str]["status"] = "failed"
                    _JOB_STATES[user_id_str]["error"] = exc.detail
            logger.warning("Could not acquire RAG indexing lease for user %s: %s", user_id_str, exc.detail)
            return
        except Exception as exc:
            with _JOB_LOCK:
                if user_id_str in _JOB_STATES:
                    _JOB_STATES[user_id_str]["status"] = "failed"
                    _JOB_STATES[user_id_str]["error"] = str(exc)
            logger.exception("Error acquiring RAG indexing lease for user %s: %s", user_id_str, exc)
            return

        try:
            while True:
                # Check for pause request
                with _JOB_LOCK:
                    current_st = _JOB_STATES.get(user_id_str, {})
                    if current_st.get("pause_requested"):
                        current_st["status"] = "paused"
                        current_st["last_checkpoint"] = datetime.now(timezone.utc).isoformat()
                        logger.info("Indexing worker for user %s paused by request.", user_id_str)
                        break

                with factory() as db:
                    # Check global AI financial budget before batch
                    try:
                        check_global_ai_budget(db)
                    except HTTPException as budget_exc:
                        with _JOB_LOCK:
                            current_st = _JOB_STATES.setdefault(user_id_str, {})
                            current_st["status"] = "budget_exceeded"
                            current_st["error"] = budget_exc.detail
                        logger.warning("RAG indexing paused for user %s: AI budget limit reached", user_id_str)
                        break

                    # Fetch next unindexed batch
                    unindexed_ids = self.get_unindexed_email_ids(db, user_id=user_id, limit=batch_size)
                    if not unindexed_ids:
                        # Mailbox indexing complete!
                        with _JOB_LOCK:
                            current_st = _JOB_STATES.setdefault(user_id_str, {})
                            current_st["status"] = "completed"
                            current_st["emails_pending"] = 0
                            current_st["completed_at"] = datetime.now(timezone.utc).isoformat()
                        # Clean up unembedded emails that could not be indexed
                        try:
                            self.purge_unembedded_emails(db, user_id=user_id)
                        except Exception:
                            pass
                        logger.info("RAG indexing job %s completed for user %s.", job_id, user_id_str)
                        break

                    # Process batch of unindexed emails
                    try:
                        res = self.pipeline.index_emails_bulk(
                            db,
                            unindexed_ids,
                            user_id=user_id,
                            force=force,
                            batch_size=50,
                        )
                        indexed = res.get("emails_indexed", 0)
                        chunks = res.get("chunks_created", 0)

                        with _JOB_LOCK:
                            current_st = _JOB_STATES.setdefault(user_id_str, {})
                            current_st["emails_processed"] += indexed
                            current_st["chunks_generated"] += chunks
                            current_st["embeddings_generated"] += chunks
                            current_st["current_batch"] += 1
                            current_st["emails_pending"] = max(0, current_st["emails_pending"] - indexed)
                            current_st["last_checkpoint"] = datetime.now(timezone.utc).isoformat()

                            # Cost accounting: ~2000 chars per chunk -> ~500 tokens
                            chunk_tokens = chunks * 500
                            cost = calculate_cost("text-embedding-3-large", chunk_tokens, 0)
                            current_st["estimated_cost_usd"] = round(current_st.get("estimated_cost_usd", 0.0) + cost, 6)

                        # Record AI usage log
                        try:
                            log_ai_usage(
                                db=db,
                                operation_type="rag_embedding",
                                model_name="text-embedding-3-large",
                                provider=getattr(settings, "embedding_provider", "azure_openai"),
                                prompt_tokens=chunks * 500,
                                completion_tokens=0,
                                ip_hash="rag-background-worker",
                                user_id=user_id,
                            )
                        except Exception:
                            pass

                        logger.info(
                            "Indexed batch #%d for user %s: %d emails, %d chunks",
                            current_st.get("current_batch", 0),
                            user_id_str,
                            indexed,
                            chunks,
                        )
                    except Exception as batch_exc:
                        logger.exception("Batch error indexing emails for user %s: %s", user_id_str, batch_exc)
                        with _JOB_LOCK:
                            current_st = _JOB_STATES.setdefault(user_id_str, {})
                            current_st["emails_failed"] += len(unindexed_ids)
                            current_st["error"] = str(batch_exc)
                        # Yield brief pause before retrying next batch
                        import time
                        time.sleep(1)

        finally:
            try:
                lease.release()
            except Exception:
                pass
            with _JOB_LOCK:
                _ACTIVE_WORKER_THREADS.pop(user_id_str, None)

    def run_indexing_sync(
        self,
        db: Session,
        user_id: UUID,
        batch_size: int = 500,
        force: bool = False,
        max_emails: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Synchronously index emails (used when synchronous execution is explicitly requested or in tests)."""
        limit = min(max_emails or batch_size, 500)
        unindexed_ids = self.get_unindexed_email_ids(db, user_id=user_id, limit=limit)
        if not unindexed_ids and not force:
            return {
                "total_emails": 0,
                "indexed_emails": 0,
                "emails_indexed": 0,
                "emails_skipped": 0,
                "chunks_created": 0,
                "total_chunks": 0,
            }

        ids_to_process = unindexed_ids if not force else list(db.scalars(
            select(Email.id).join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(GmailAccount.user_id == user_id)
            .limit(limit)
        ).all())

        results = self.pipeline.index_emails_bulk(
            db,
            ids_to_process,
            user_id=user_id,
            force=force,
            batch_size=50,
        )
        return results

    def search_rag(self, query: str, user_id: UUID, top_k: int = 5,
                   filter_sender: Optional[str] = None) -> List[Dict[str, Any]]:
        self.vector_store.db_session = self.vector_store.db_session or None
        filter_dict = {"sender_email": filter_sender} if filter_sender else None
        from app.database.connection import SessionLocal
        with SessionLocal() as db:
            manager = PostgresVectorStoreManager(
                embedding_function=self.vector_store.embedding_function,
                db_session=db,
            )
            manager.collection_name = self.collection_name
            rows = manager.search_with_scores(query, user_id, top_k=top_k, filter_dict=filter_dict)
        return [{
            "record_id": doc.metadata.get("chunk_id"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "email_id": doc.metadata.get("email_id"),
            "user_id": doc.metadata.get("user_id"),
            "gmail_message_id": doc.metadata.get("gmail_message_id"),
            "subject": doc.metadata.get("subject"),
            "sender_id": doc.metadata.get("sender_email"),
            "sender_email": doc.metadata.get("sender_email"),
            "recipient_email": doc.metadata.get("recipient_email"),
            "received_date": doc.metadata.get("received_date"),
            "similarity_score": score,
            "document": doc.metadata.get("embedding_text") or doc.page_content,
        } for doc, score in rows]


default_rag_service = RAGIndexerService()

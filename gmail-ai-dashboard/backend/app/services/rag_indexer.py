"""PostgreSQL RAG indexing service."""

from __future__ import annotations

import logging

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.developer_settings import database_url_context
from app.models.email import Email, GmailAccount, User
from app.models.rag import EmailChunk, EmailEmbedding
from app.rag.indexing import RAGIndexingPipeline
from app.rag.pgvector_store import PostgresVectorStoreManager


logger = logging.getLogger(__name__)


class _PostgresCollectionFacade:
    """Small compatibility surface for callers that used ``collection.count``."""

    def __init__(self, manager: PostgresVectorStoreManager):
        self.manager = manager

    def count(self, user_id=None) -> int:
        return self.manager.count(user_id=user_id)


class RAGIndexerService:
    """Compatibility API backed entirely by ``email_embeddings``."""

    def __init__(self, collection_name=None, embedding_provider=None):
        self.collection_name = collection_name or "postgres_halfvec_3072"
        self.embedding_provider = embedding_provider
        self.vector_store = PostgresVectorStoreManager(embedding_function=embedding_provider)
        self.vector_store.collection_name = self.collection_name
        self.pipeline = RAGIndexingPipeline(vector_store_manager=self.vector_store)
        self.collection = _PostgresCollectionFacade(self.vector_store)

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
        # Purge any remaining unembedded emails when finishing batch
        if not batch_size or len(ids) < batch_size:
            self.purge_unembedded_emails(db, user_id=user_id)
        return {
            "total_unindexed": len(ids),
            "emails_indexed": bulk_res["emails_indexed"],
            "chunks_created": bulk_res["chunks_created"],
        }

    def purge_unembedded_emails(self, db: Session, user_id: Optional[UUID] = None) -> int:
        return self.pipeline.purge_unembedded_emails(db, user_id=user_id)

    def search_rag(self, query: str, user_id: UUID, top_k: int = 5,
                   filter_sender: Optional[str] = None) -> List[Dict[str, Any]]:
        self.vector_store.db_session = self.vector_store.db_session or None
        filter_dict = {"sender_email": filter_sender} if filter_sender else None
        # SessionLocal is used here for compatibility; normal request paths
        # use the retriever with the request's session.
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

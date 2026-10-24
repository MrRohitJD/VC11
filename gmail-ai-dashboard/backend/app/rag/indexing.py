"""Idempotent PostgreSQL/pgvector email indexing.

Email chunks, vectors, and searchable envelope metadata are written in one
transaction to ``email_chunks`` and ``email_embeddings``.  There is no local
vector-store side effect.
"""

from __future__ import annotations

import logging
from email.utils import getaddresses
from typing import Optional
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.email import Email, GmailAccount
from app.models.rag import EmailChunk, EmailEmbedding
from app.rag.chunking import default_chunker
from app.rag.embeddings import get_embedding_model
from app.services.email_chunker import create_email_chunks

logger = logging.getLogger(__name__)


def _recipient_list(email: Email) -> list[str]:
    values = [email.to_email or "", email.cc or "", email.bcc or ""]
    return [address for _, address in getaddresses([value for value in values if value]) if address]


class RAGIndexingPipeline:
    """Generate and persist 3072-dimensional vectors in PostgreSQL."""

    def __init__(self, vector_store_manager=None, chunker=default_chunker, chroma_manager=None):
        # ``chroma_manager`` is a compatibility keyword only. The supplied
        # manager must expose an embedding_function; no Chroma API is called.
        self.vector_store = vector_store_manager or chroma_manager
        if self.vector_store is None:
            from app.rag.pgvector_store import PostgresVectorStoreManager
            self.vector_store = PostgresVectorStoreManager()
        self.chunker = chunker
        self.embedding_function = getattr(self.vector_store, "embedding_function", None) or get_embedding_model()

    @property
    def chroma(self):
        """Deprecated alias retained for integrations that only read the manager."""
        return self.vector_store

    def index_email(
        self,
        db: Session,
        email_id: UUID | str,
        user_id: Optional[UUID | str] = None,
        force: bool = False,
    ) -> int:
        email_uuid = UUID(str(email_id))
        email = db.scalar(
            select(Email)
            .where(Email.id == email_uuid)
        )
        if email is None:
            logger.warning("Email %s not found for RAG indexing", email_id)
            return 0

        # Indexing is an account-owned operation.  The authenticated user is
        # normally passed by ``index_user_emails``; validate it here too so a
        # caller cannot accidentally index another user's email through this
        # lower-level API.
        account = db.scalar(
            select(GmailAccount).where(GmailAccount.id == email.gmail_account_id)
        )
        if account is None:
            logger.warning("Email %s has no owning Gmail account", email_id)
            return 0
        if user_id is not None and UUID(str(user_id)) != account.user_id:
            raise PermissionError("Email does not belong to the requested user")

        records = create_email_chunks(
            subject=email.subject,
            body=email.body,
            body_html=email.body_html,
            sender_name=email.sender_name,
            sender_email=email.from_email,
            gmail_message_id=email.gmail_message_id,
            received_at=email.received_at,
            recipient_emails=email.to_email,
        )
        if not records:
            self.delete_email(db, email_uuid)
            return 0

        existing = list(db.scalars(
            select(EmailChunk).where(EmailChunk.email_id == email_uuid).order_by(EmailChunk.chunk_index)
        ))
        hashes = [record.content_hash for record in records]
        existing_hashes = [chunk.content_hash for chunk in existing]
        dimensions = int(getattr(self.embedding_function, "dimensions", settings.embedding_dimensions))
        indexed_count = db.scalar(select(func.count(EmailEmbedding.id)).where(
            EmailEmbedding.chunk_id.in_([chunk.id for chunk in existing]) if existing else False,
            EmailEmbedding.status == "INDEXED",
            EmailEmbedding.embedding.is_not(None),
            EmailEmbedding.embedding_dimensions == settings.embedding_dimensions,
        )) or 0
        if not force and existing_hashes == hashes and len(existing) == len(records) and indexed_count == len(records):
            return len(existing)

        if dimensions != settings.embedding_dimensions:
            raise ValueError(
                f"Configured PostgreSQL halfvec is {settings.embedding_dimensions} dimensions; provider returned {dimensions}"
            )
        vectors = self.embedding_function.embed_documents([record.embedding_text for record in records])
        if len(vectors) != len(records) or any(len(vector) != settings.embedding_dimensions for vector in vectors):
            raise ValueError(f"Embedding provider must return {settings.embedding_dimensions}-dimensional vectors")

        # Cascading deletes remove the old metadata/vector rows before the new
        # rows are inserted, keeping one current vector per chunk.
        db.execute(delete(EmailChunk).where(EmailChunk.email_id == email_uuid))
        db.flush()
        recipients = _recipient_list(email)
        provider = getattr(self.embedding_function, "provider", settings.embedding_provider)
        model = getattr(
            self.embedding_function,
            "deployment_name",
            getattr(self.embedding_function, "model", settings.embedding_model_name),
        )
        for index, (record, vector) in enumerate(zip(records, vectors)):
            chunk = EmailChunk(
                email_id=email_uuid,
                chunk_index=index,
                chunk_text=record.chunk_text,
                start_index=record.start_index,
                end_index=record.end_index,
                chunk_strategy=record.chunk_strategy,
                content_hash=record.content_hash,
            )
            db.add(chunk)
            db.flush()
            db.add(EmailEmbedding(
                chunk_id=chunk.id,
                email_id=email_uuid,
                email_subject=email.subject or "",
                sender_email=email.from_email or "",
                recipient_emails=recipients,
                email_sent_at=email.received_at,
                embedding_text=record.embedding_text,
                embedding_type="email",
                # These two legacy-named columns are retained for schema
                # compatibility; their values identify PostgreSQL records.
                chroma_collection=self.vector_store.collection_name,
                chroma_record_id=f"pgvector_{chunk.id}",
                embedding_provider=str(provider),
                embedding_model=str(model),
                embedding_dimensions=settings.embedding_dimensions,
                embedding=list(vector),
                content_hash=record.content_hash,
                status="INDEXED",
            ))
        db.commit()
        logger.info("Indexed %d PostgreSQL vector chunk(s) for email %s", len(records), email_id)
        return len(records)

    def index_user_emails(self, db: Session, user_id: UUID | str, limit: Optional[int] = None,
                          batch_size: Optional[int] = None, force: bool = False) -> dict[str, int]:
        user_uuid = UUID(str(user_id))
        query = (select(Email.id).join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
                 .where(GmailAccount.user_id == user_uuid)
                 .order_by(Email.received_at.desc().nullslast()))
        effective_limit = limit or batch_size
        if effective_limit:
            query = query.limit(effective_limit)
        email_ids = db.scalars(query).all()
        indexed_count = 0
        chunks = 0
        for email_id in email_ids:
            try:
                count = self.index_email(db, email_id, user_id=user_uuid, force=force)
                if count:
                    indexed_count += 1
                    chunks += count
            except Exception:
                db.rollback()
                logger.exception("Error indexing email %s", email_id)
        return {
            "total_emails": len(email_ids),
            "indexed_emails": indexed_count,
            "emails_indexed": indexed_count,
            "emails_skipped": max(0, len(email_ids) - indexed_count),
            "chunks_created": chunks,
            "total_chunks": chunks,
        }

    def delete_email(self, db: Session, email_id: UUID | str) -> None:
        db.execute(delete(EmailChunk).where(EmailChunk.email_id == UUID(str(email_id))))
        db.commit()


default_indexing_pipeline = RAGIndexingPipeline()
default_pipeline = default_indexing_pipeline

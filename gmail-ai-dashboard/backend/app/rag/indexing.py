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

    def __init__(self, vector_store_manager=None, chunker=default_chunker):
        self.vector_store = vector_store_manager
        if self.vector_store is None:
            from app.rag.pgvector_store import PostgresVectorStoreManager
            self.vector_store = PostgresVectorStoreManager()
        self.chunker = chunker
        self.embedding_function = getattr(self.vector_store, "embedding_function", None) or get_embedding_model()

    def index_email(
        self,
        db: Session,
        email_id: UUID | str,
        user_id: Optional[UUID | str] = None,
        force: bool = False,
    ) -> int:
        email_uuid = UUID(str(email_id))
        email = db.scalar(select(Email).where(Email.id == email_uuid))
        if email is None:
            logger.warning("Email %s not found for RAG indexing", email_id)
            return 0

        records = create_email_chunks(
            subject=email.subject,
            body=email.body,
            body_html=email.body_html,
            snippet=email.snippet,
            sender_name=email.sender_name,
            sender_email=email.from_email,
            gmail_message_id=email.gmail_message_id,
            received_at=email.received_at,
            recipient_emails=email.to_email,
        )
        if not records:
            self.delete_email(db, email_uuid)
            db.execute(delete(Email).where(Email.id == email_uuid))
            db.commit()
            logger.info("Purged unembeddable email %s from database", email_id)
            return 0

        existing = list(db.scalars(
            select(EmailChunk).where(EmailChunk.email_id == email_uuid).order_by(EmailChunk.chunk_index)
        ))
        hashes = [record.content_hash for record in records]
        existing_hashes = [chunk.content_hash for chunk in existing]
        raw_dim = getattr(self.embedding_function, "dimensions", None)
        dimensions = int(raw_dim) if raw_dim is not None else settings.embedding_dimensions
        indexed_count = db.scalar(select(func.count(EmailEmbedding.id)).where(
            EmailEmbedding.chunk_id.in_([chunk.id for chunk in existing]) if existing else False,
            EmailEmbedding.status == "INDEXED",
            EmailEmbedding.embedding.is_not(None),
            EmailEmbedding.embedding_dimensions == settings.embedding_dimensions,
        )) or 0
        if not force and existing_hashes == hashes and len(existing) == len(records) and indexed_count == len(records):
            return len(existing)

        try:
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
                    vector_collection=self.vector_store.collection_name,
                    vector_record_id=f"pgvector_{chunk.id}",
                    embedding_provider=str(provider),
                    embedding_model=str(model),
                    embedding_dimensions=settings.embedding_dimensions,
                    embedding=list(vector),
                    content_hash=record.content_hash,
                    status="INDEXED",
                ))
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.warning("Failed to embed email %s: %s; purging unembedded email from database", email_id, exc)
            try:
                self.delete_email(db, email_uuid)
                db.execute(delete(Email).where(Email.id == email_uuid))
                db.commit()
            except Exception:
                db.rollback()
            raise
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
                logger.exception("Error indexing email %s; purging unembedded email", email_id)
                try:
                    db.execute(delete(Email).where(Email.id == email_id))
                    db.commit()
                except Exception:
                    db.rollback()
        purged = 0
        if not effective_limit:
            purged = self.purge_unembedded_emails(db, user_id=user_uuid)
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

    def purge_unembedded_emails(self, db: Session, user_id: Optional[UUID | str] = None) -> int:
        """Purge any email in the database that has not been embedded with pgvector embeddings."""
        from sqlalchemy import exists
        indexed = (select(1).select_from(EmailChunk)
                   .join(EmailEmbedding, EmailEmbedding.chunk_id == EmailChunk.id)
                   .where(EmailChunk.email_id == Email.id,
                          EmailEmbedding.status == "INDEXED",
                          EmailEmbedding.embedding.is_not(None),
                          EmailEmbedding.embedding_dimensions == settings.embedding_dimensions))
        stmt = select(Email.id).join(GmailAccount, GmailAccount.id == Email.gmail_account_id).where(~exists(indexed))
        if user_id:
            stmt = stmt.where(GmailAccount.user_id == UUID(str(user_id)))
        unembedded_ids = list(db.scalars(stmt).all())
        if not unembedded_ids:
            return 0
        db.execute(delete(Email).where(Email.id.in_(unembedded_ids)))
        db.commit()
        logger.info("Purged %d unembedded emails from database", len(unembedded_ids))
        return len(unembedded_ids)


default_indexing_pipeline = RAGIndexingPipeline()
default_pipeline = default_indexing_pipeline

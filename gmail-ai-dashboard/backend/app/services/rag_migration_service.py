"""Batch migration utility for PostgreSQL pgvector embeddings.

This replaces the old Chroma v1/v2 migration. Existing ``email_chunks`` are
re-embedded into ``email_embeddings`` and all metadata is written in the same
PostgreSQL row as the vector.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from email.utils import getaddresses
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.email import Email
from app.models.rag import EmailChunk, EmailEmbedding
from app.rag.pgvector_store import PostgresVectorStoreManager
from app.services.email_chunker import format_embedding_text
from app.services.embedding_provider import EmbeddingProvider, default_embedding_provider

logger = logging.getLogger(__name__)


@dataclass
class MigrationProgress:
    total_chunks: int
    already_indexed: int
    migrated: int
    failed: int
    percent_complete: float
    elapsed_seconds: float


@dataclass
class ComparisonResult:
    query: str
    v1_collection: str
    v2_collection: str
    v1_results: List[Dict[str, Any]]
    v2_results: List[Dict[str, Any]]
    v1_latency_ms: float
    v2_latency_ms: float


def _recipients(email: Email) -> list[str]:
    values = [email.to_email or "", email.cc or "", email.bcc or ""]
    return [address for _, address in getaddresses([value for value in values if value]) if address]


class RAGMigrationService:
    """Re-embed chunks into PostgreSQL, with resumable batch progress."""

    def __init__(self, chroma_client=None, embedding_provider: Optional[EmbeddingProvider] = None,
                 target_collection_name: Optional[str] = None):
        self.target_collection = target_collection_name or "postgres_halfvec_3072"
        self.embedder = embedding_provider or default_embedding_provider

    def migrate_all_chunks(self, db: Session, batch_size: int = 50,
                           progress_callback: Optional[Callable[[MigrationProgress], None]] = None) -> MigrationProgress:
        started = time.time()
        chunks = db.scalars(select(EmailChunk).order_by(EmailChunk.created_at)).all()
        total = len(chunks)
        indexed_ids = set(db.scalars(select(EmailEmbedding.chunk_id).where(
            EmailEmbedding.chroma_collection == self.target_collection,
            EmailEmbedding.status == "INDEXED",
            EmailEmbedding.embedding.is_not(None),
            EmailEmbedding.embedding_dimensions == settings.embedding_dimensions,
        )).all())
        pending = [chunk for chunk in chunks if chunk.id not in indexed_ids]
        migrated = failed = 0

        emails = {email.id: email for email in db.scalars(
            select(Email).where(Email.id.in_({chunk.email_id for chunk in pending}))
        ).all()} if pending else {}
        provider = getattr(self.embedder, "provider", settings.embedding_provider)
        model = getattr(self.embedder, "deployment_name", getattr(self.embedder, "model_name", settings.embedding_model_name))

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            try:
                texts = []
                for chunk in batch:
                    email = emails.get(chunk.email_id)
                    texts.append(format_embedding_text(
                        subject=email.subject if email else "",
                        chunk_text=chunk.chunk_text,
                        sender_name=email.sender_name if email else None,
                        sender_email=email.from_email if email else None,
                        gmail_message_id=email.gmail_message_id if email else None,
                        received_at=email.received_at if email else None,
                        recipient_emails=email.to_email if email else None,
                    ))
                vectors = self.embedder.embed_documents(texts)
                if any(len(vector) != settings.embedding_dimensions for vector in vectors):
                    raise ValueError(f"Embedding provider must return {settings.embedding_dimensions}-dimensional vectors")
                for chunk, text, vector in zip(batch, texts, vectors):
                    email = emails.get(chunk.email_id)
                    embedding = db.scalar(select(EmailEmbedding).where(
                        EmailEmbedding.chunk_id == chunk.id,
                        EmailEmbedding.chroma_collection == self.target_collection,
                    ))
                    if embedding is None:
                        embedding = EmailEmbedding(
                            chunk_id=chunk.id,
                            chroma_collection=self.target_collection,
                            chroma_record_id=f"pgvector_{chunk.id}",
                        )
                        db.add(embedding)
                    embedding.email_id = chunk.email_id
                    embedding.email_subject = email.subject if email else ""
                    embedding.sender_email = email.from_email if email else ""
                    embedding.recipient_emails = _recipients(email) if email else []
                    embedding.email_sent_at = email.received_at if email else None
                    embedding.embedding_text = text
                    embedding.embedding_type = "email"
                    embedding.embedding_provider = str(provider)
                    embedding.embedding_model = str(model)
                    embedding.embedding_dimensions = settings.embedding_dimensions
                    embedding.embedding = list(vector)
                    embedding.content_hash = chunk.content_hash
                    embedding.status = "INDEXED"
                db.commit()
                migrated += len(batch)
            except Exception as exc:
                db.rollback()
                failed += len(batch)
                logger.error("Failed PostgreSQL embedding migration batch at %d: %s", start, exc)

            progress = MigrationProgress(
                total_chunks=total,
                already_indexed=len(indexed_ids),
                migrated=migrated,
                failed=failed,
                percent_complete=round(((len(indexed_ids) + migrated + failed) / total) * 100, 1) if total else 100.0,
                elapsed_seconds=round(time.time() - started, 2),
            )
            if progress_callback:
                progress_callback(progress)

        return MigrationProgress(
            total_chunks=total,
            already_indexed=len(indexed_ids),
            migrated=migrated,
            failed=failed,
            percent_complete=100.0,
            elapsed_seconds=round(time.time() - started, 2),
        )

    def compare_retrieval(self, db: Session, query: str, user_id: UUID, top_k: int = 5,
                          v1_collection: str = "gmail_email_chunks",
                          v2_collection: Optional[str] = None) -> ComparisonResult:
        def search(collection: str) -> tuple[List[Dict[str, Any]], float]:
            started = time.perf_counter()
            manager = PostgresVectorStoreManager(embedding_function=self.embedder, db_session=db)
            manager.collection_name = collection
            docs = manager.search_with_scores(query, user_id, top_k=top_k,
                                              filter_dict={"chroma_collection": collection})
            return [{"email_id": doc.metadata.get("email_id"), "subject": doc.metadata.get("subject"),
                     "document": doc.page_content, "score": score} for doc, score in docs], (time.perf_counter() - started) * 1000

        v2 = v2_collection or self.target_collection
        first, first_ms = search(v1_collection)
        second, second_ms = search(v2)
        return ComparisonResult(query, v1_collection, v2, first, second, first_ms, second_ms)

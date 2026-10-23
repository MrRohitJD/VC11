"""Incremental, idempotent RAG indexing pipeline.

Coordinates:
- Gmail / PostgreSQL email retrieval
- LangChain document conversion and cleaning
- RecursiveCharacterTextSplitter chunking
- Azure OpenAI embedding generation
- ChromaDB vector persistence
- PostgreSQL chunk and embedding audit records (email_chunks, email_embeddings)
- Idempotency via SHA-256 content hashes (skips unchanged emails)
- Obsolete chunk cleanup on updates and deletions
"""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models.email import Email, GmailAccount
from app.models.rag import EmailChunk, EmailEmbedding
from app.rag.chunking import default_chunker
from app.rag.loaders import GmailDocumentAdapter
from app.rag.vectorstore import default_chroma_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGIndexingPipeline:
    """Manages incremental indexing of emails into ChromaDB and PostgreSQL."""

    def __init__(self, chroma_manager=default_chroma_manager, chunker=default_chunker):
        self.chroma = chroma_manager
        self.chunker = chunker

    def index_email(
        self,
        db: Session,
        email_id: UUID | str,
        user_id: Optional[UUID | str] = None,
        force: bool = False,
    ) -> int:
        """Idempotently chunk, embed, and index a single email.

        Returns the number of indexed chunks.
        """
        # 1. Fetch Email
        email_uuid = UUID(str(email_id))
        email = db.scalar(select(Email).where(Email.id == email_uuid))
        if not email:
            logger.warning("Email %s not found for RAG indexing", email_id)
            return 0

        # Resolve user_id
        resolved_user_id = user_id
        if resolved_user_id is None:
            account = db.scalar(select(GmailAccount).where(GmailAccount.id == email.gmail_account_id))
            if account:
                resolved_user_id = account.user_id

        # 2. Convert to LangChain Document
        raw_doc = GmailDocumentAdapter.email_to_document(email, user_id=resolved_user_id)

        # 3. Split using LangChain RecursiveCharacterTextSplitter
        chunk_docs = self.chunker.split_document(raw_doc)
        if not chunk_docs:
            self.delete_email(db, email_uuid)
            return 0

        if settings.vector_backend == "postgres":
            return self._index_email_postgres(db, email_uuid, chunk_docs, force=force)

        # 4. Idempotency Check: Compare hashes with existing chunks in PostgreSQL
        existing_chunks = db.scalars(
            select(EmailChunk).where(EmailChunk.email_id == email_uuid).order_by(EmailChunk.chunk_index)
        ).all()

        current_hashes = [doc.metadata.get("content_hash", "") for doc in chunk_docs]
        existing_hashes = [c.content_hash for c in existing_chunks]

        indexed_embedding_count = db.scalar(
            select(func.count(EmailEmbedding.id))
            .where(
                EmailEmbedding.chunk_id.in_([chunk.id for chunk in existing_chunks]),
                EmailEmbedding.chroma_collection == self.chroma.collection_name,
                EmailEmbedding.status == "INDEXED",
            )
        ) or 0

        if (
            not force
            and current_hashes == existing_hashes
            and len(existing_chunks) == len(chunk_docs)
            and indexed_embedding_count == len(chunk_docs)
        ):
            # No changes detected, skip re-embedding
            logger.debug("Email %s chunks unchanged. Skipping indexing.", email_id)
            return len(existing_chunks)

        # 5. Remove any obsolete chunks from ChromaDB & PostgreSQL
        self.delete_email(db, email_uuid)

        # 6. Index into ChromaDB through LangChain
        chroma_ids = self.chroma.add_chunks(chunk_docs)

        # 7. Persist chunk and embedding records in PostgreSQL
        for idx, doc in enumerate(chunk_docs):
            chunk_rec = EmailChunk(
                email_id=email_uuid,
                chunk_index=idx,
                chunk_text=doc.page_content,
                start_index=0,
                end_index=len(doc.page_content),
                chunk_strategy=doc.metadata.get("chunk_strategy", "recursive_character"),
                content_hash=doc.metadata.get("content_hash", ""),
            )
            db.add(chunk_rec)
            db.flush()

            embedding_rec = EmailEmbedding(
                chunk_id=chunk_rec.id,
                chroma_collection=self.chroma.collection_name,
                chroma_record_id=chroma_ids[idx],
                embedding_provider="azure_openai",
                embedding_model="text-embedding-3-large",
                embedding_dimensions=3072,
                content_hash=doc.metadata.get("content_hash", ""),
                status="INDEXED",
            )
            db.add(embedding_rec)

        db.commit()
        logger.info("Indexed %d chunks for email %s", len(chunk_docs), email_id)
        return len(chunk_docs)

    def _index_email_postgres(self, db: Session, email_id: UUID, chunk_docs, *, force: bool) -> int:
        """Write chunks and vectors in one PostgreSQL transaction."""
        existing = list(db.scalars(select(EmailChunk).where(EmailChunk.email_id == email_id)
                                   .order_by(EmailChunk.chunk_index)))
        hashes = [doc.metadata.get("content_hash", "") for doc in chunk_docs]
        if not force and [chunk.content_hash for chunk in existing] == hashes and existing:
            valid = db.scalar(select(func.count(EmailEmbedding.id)).where(
                EmailEmbedding.chunk_id.in_([chunk.id for chunk in existing]),
                EmailEmbedding.status == "INDEXED", EmailEmbedding.embedding.is_not(None),
                EmailEmbedding.embedding_dimensions == 3072,
            )) or 0
            if valid == len(existing):
                return len(existing)

        vectors = self.chroma.embedding_function.embed_documents([doc.page_content for doc in chunk_docs])
        if any(len(vector) != 3072 for vector in vectors):
            raise ValueError("Production indexing requires 3072-dimensional embeddings")
        db.execute(delete(EmailChunk).where(EmailChunk.email_id == email_id))
        db.flush()
        for idx, (doc, vector) in enumerate(zip(chunk_docs, vectors)):
            chunk = EmailChunk(
                email_id=email_id, chunk_index=idx, chunk_text=doc.page_content,
                start_index=0, end_index=len(doc.page_content),
                chunk_strategy=doc.metadata.get("chunk_strategy", "recursive_character"),
                content_hash=hashes[idx],
            )
            db.add(chunk)
            db.flush()
            recipients = [value.strip() for value in (email.to_email or "").split(",") if value.strip()]
            db.add(EmailEmbedding(
                chunk_id=chunk.id, email_id=email.id,
                email_subject=email.subject or "", sender_email=email.from_email or "",
                recipient_emails=recipients, email_sent_at=email.received_at,
                embedding_text=doc.page_content, embedding_type="email",
                chroma_collection=self.chroma.collection_name,
                chroma_record_id=str(chunk.id), embedding=list(vector),
                embedding_provider=settings.embedding_provider,
                embedding_model=settings.embedding_model_name,
                embedding_dimensions=3072, content_hash=hashes[idx], status="INDEXED",
            ))
        db.commit()
        return len(chunk_docs)

    def index_user_emails(
        self,
        db: Session,
        user_id: UUID | str,
        limit: Optional[int] = None,
        batch_size: Optional[int] = None,
        force: bool = False,
    ) -> dict[str, int]:
        """Index all emails belonging to a user incrementally."""
        user_uuid = UUID(str(user_id))
        query = (
            select(Email.id)
            .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(GmailAccount.user_id == user_uuid)
            .order_by(Email.received_at.desc().nullslast())
        )
        effective_limit = limit or batch_size
        if effective_limit:
            query = query.limit(effective_limit)

        email_ids = db.scalars(query).all()
        indexed_count = 0
        total_chunks = 0

        for eid in email_ids:
            try:
                chunks = self.index_email(db, eid, user_id=user_uuid, force=force)
                if chunks > 0:
                    indexed_count += 1
                    total_chunks += chunks
            except Exception as exc:
                logger.error("Error indexing email %s: %s", eid, exc)

        return {
            "total_emails": len(email_ids),
            "indexed_emails": indexed_count,
            "emails_indexed": indexed_count,
            "emails_skipped": max(0, len(email_ids) - indexed_count),
            "chunks_created": total_chunks,
            "total_chunks": total_chunks,
        }

    def delete_email(self, db: Session, email_id: UUID | str) -> None:
        """Remove all chunks of an email from ChromaDB and PostgreSQL."""
        email_uuid = UUID(str(email_id))
        if settings.vector_backend != "postgres":
            self.chroma.delete_by_email_id(email_uuid)
        db.execute(delete(EmailChunk).where(EmailChunk.email_id == email_uuid))
        db.commit()


default_indexing_pipeline = RAGIndexingPipeline()
default_pipeline = default_indexing_pipeline

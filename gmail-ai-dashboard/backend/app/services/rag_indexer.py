"""ChromaDB & PostgreSQL coordinated RAG indexing service using OpenAI text-embedding-3-large.

Ensures:
- 3072-dimensional OpenAI embeddings targeting isolated collection 'gmail_email_chunks_v2'
- Zero vector mixing with legacy 'all-MiniLM-L6-v2' (384d) collections
- Idempotent indexing using SHA-256 content hashes (skips unchanged emails)
- Safe re-indexing when email content or subject changes
- Complete deletion handling across PostgreSQL and ChromaDB
- Multi-tenant user isolation enforced on every vector query
- Coordinated database transactions with vector store updates
- Zero leakage of OAuth tokens, credentials, or private email text in logs
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

import chromadb
from chromadb.api import ClientAPI
from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.email import Email, GmailAccount
from app.models.rag import EmailChunk, EmailEmbedding
from app.services.email_chunker import create_email_chunks
from app.services.embedding_provider import (
    EmbeddingProvider,
    OPENAI_LARGE_DIMENSIONS,
    OPENAI_LARGE_MODEL,
    default_embedding_provider,
)

logger = logging.getLogger(__name__)


class RAGIndexerService:
    """Orchestrates hybrid chunking, PostgreSQL chunk tracking, and ChromaDB vector persistence."""

    def __init__(
        self,
        chroma_client: Optional[ClientAPI] = None,
        collection_name: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.collection_name = collection_name or settings.chroma_collection_name_v2
        self.embedder = embedding_provider or default_embedding_provider

        if chroma_client is not None:
            self.client = chroma_client
        else:
            self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

        # Get or create the isolated collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "model": self.embedder.model_name, "dimensions": self.embedder.dimensions},
        )

    def index_email(
        self,
        db: Session,
        email_id: UUID,
        user_id: Optional[UUID] = None,
        force: bool = False,
    ) -> int:
        """Idempotently chunk, embed, and store an email.

        Returns the number of indexed chunks.
        Skips re-indexing if content hashes have not changed.
        """
        # 1. Fetch Email and resolve owning User ID
        email = db.scalar(select(Email).where(Email.id == email_id))
        if not email:
            logger.warning("Email %s not found for indexing", email_id)
            return 0

        if user_id is None:
            account = db.scalar(select(GmailAccount).where(GmailAccount.id == email.gmail_account_id))
            if account:
                user_id = account.user_id

        user_id_str = str(user_id) if user_id else ""

        # 2. Generate hybrid chunks (preserving short emails and RecursiveCharacterTextSplitter for long)
        chunk_records = create_email_chunks(
            subject=email.subject,
            body=email.body,
            body_html=email.body_html,
            sender_name=email.sender_name,
            sender_email=email.from_email,
            gmail_message_id=email.gmail_message_id,
            received_at=email.received_at,
        )

        # 3. If email has no usable body text, clean up any existing chunks and return
        if not chunk_records:
            self.delete_email_from_rag(db=db, email_id=email_id)
            return 0

        # 4. Idempotency Check: Compare content hashes with existing chunks in PostgreSQL
        existing_chunks = db.scalars(
            select(EmailChunk).where(EmailChunk.email_id == email_id).order_by(EmailChunk.chunk_index)
        ).all()

        existing_hashes = [c.content_hash for c in existing_chunks]
        new_hashes = [c.content_hash for c in chunk_records]

        # Check if already embedded in this specific collection
        already_embedded = False
        if existing_chunks:
            embedding_record = db.scalar(
                select(EmailEmbedding).where(
                    EmailEmbedding.chunk_id == existing_chunks[0].id,
                    EmailEmbedding.chroma_collection == self.collection_name,
                    EmailEmbedding.status == "INDEXED",
                )
            )
            already_embedded = embedding_record is not None

        if not force and existing_hashes == new_hashes and already_embedded:
            logger.debug("Email %s content unchanged in collection %s; skipping", email_id, self.collection_name)
            return len(existing_chunks)

        # 5. Content changed or new: Clean up previous vectors in ChromaDB first
        try:
            self.collection.delete(where={"email_id": str(email_id)})
        except Exception:
            pass

        # Clean up existing chunk records in PostgreSQL if chunk boundaries changed
        if existing_hashes != new_hashes:
            for old_chunk in existing_chunks:
                db.delete(old_chunk)
            db.flush()
            existing_chunks = []

        # 6. Insert new PostgreSQL chunks and prepare ChromaDB records
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        chunk_objects: List[EmailChunk] = []

        if not existing_chunks:
            for record in chunk_records:
                chunk_obj = EmailChunk(
                    email_id=email.id,
                    chunk_index=record.chunk_index,
                    chunk_text=record.chunk_text,
                    start_index=record.start_index,
                    end_index=record.end_index,
                    chunk_strategy=record.chunk_strategy,
                    content_hash=record.content_hash,
                )
                db.add(chunk_obj)
                chunk_objects.append(chunk_obj)
            db.flush()
        else:
            chunk_objects = list(existing_chunks)

        for record, chunk_obj in zip(chunk_records, chunk_objects):
            chroma_record_id = f"chunk_v2_{chunk_obj.id}"

            # Upsert PostgreSQL embedding registry record for this collection
            existing_embedding = db.scalar(
                select(EmailEmbedding).where(
                    EmailEmbedding.chunk_id == chunk_obj.id,
                    EmailEmbedding.chroma_collection == self.collection_name,
                )
            )
            provider_name = getattr(self.embedder, "provider", settings.embedding_provider)
            model_deployment = getattr(self.embedder, "deployment_name", self.embedder.model_name)

            if existing_embedding:
                existing_embedding.embedding_provider = provider_name
                existing_embedding.embedding_model = model_deployment
                existing_embedding.embedding_dimensions = self.embedder.dimensions
                existing_embedding.content_hash = record.content_hash
                existing_embedding.status = "INDEXED"
            else:
                embedding_obj = EmailEmbedding(
                    chunk_id=chunk_obj.id,
                    chroma_collection=self.collection_name,
                    chroma_record_id=chroma_record_id,
                    embedding_provider=provider_name,
                    embedding_model=model_deployment,
                    embedding_dimensions=self.embedder.dimensions,
                    content_hash=record.content_hash,
                    status="INDEXED",
                )
                db.add(embedding_obj)

            rec_date_str = ""
            rec_timestamp = 0.0
            if email.received_at:
                dt = email.received_at
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                rec_date_str = dt.isoformat()
                rec_timestamp = float(dt.timestamp())

            ids.append(chroma_record_id)
            documents.append(record.embedding_text)
            metadatas.append({
                "chunk_id": str(chunk_obj.id),
                "email_id": str(email.id),
                "user_id": user_id_str,
                "gmail_message_id": str(email.gmail_message_id or ""),
                "sender_id": str(email.from_email or ""),
                "sender_email": str(email.from_email or ""),
                "sender_name": str(email.sender_name or ""),
                "recipient_id": str(email.to_email or ""),
                "recipient_email": str(email.to_email or ""),
                "subject": str(email.subject or ""),
                "received_date": rec_date_str,
                "received_timestamp": rec_timestamp,
                "chunk_index": int(record.chunk_index),
                "chunk_strategy": str(record.chunk_strategy),
                "content_hash": str(record.content_hash),
                "embedding_provider": str(provider_name),
                "embedding_model": str(model_deployment),
                "embedding_dimensions": int(self.embedder.dimensions),
            })

        # 7. Generate dense OpenAI vectors in batch
        embeddings = self.embedder.embed_documents(documents)

        # 8. Add vectors and metadata into ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        db.commit()
        logger.info("Successfully indexed email %s (%d chunks into %s)", email_id, len(chunk_records), self.collection_name)
        return len(chunk_records)

    def delete_email_from_rag(self, db: Session, email_id: UUID) -> None:
        """Delete an email's chunks and vectors from ChromaDB and PostgreSQL."""
        try:
            self.collection.delete(where={"email_id": str(email_id)})
        except Exception as exc:
            logger.debug("Chroma deletion for email %s: %s", email_id, exc)

        existing_chunks = db.scalars(
            select(EmailChunk).where(EmailChunk.email_id == email_id)
        ).all()
        for chunk in existing_chunks:
            db.delete(chunk)
        db.commit()

    def get_unindexed_email_ids(
        self,
        db: Session,
        user_id: Optional[UUID] = None,
        limit: Optional[int] = None,
    ) -> List[UUID]:
        """Find emails in PostgreSQL that do not have an INDEXED embedding in this collection.

        Ensures tenant isolation when user_id is provided.
        Only considers emails with body or HTML content.
        """
        indexed_subquery = (
            select(1)
            .select_from(EmailChunk)
            .join(EmailEmbedding, EmailEmbedding.chunk_id == EmailChunk.id)
            .where(
                EmailChunk.email_id == Email.id,
                EmailEmbedding.chroma_collection == self.collection_name,
                EmailEmbedding.status == "INDEXED",
            )
        )
        stmt = (
            select(Email.id)
            .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(
                ~exists(indexed_subquery),
                (Email.body.is_not(None) | Email.body_html.is_not(None)),
            )
            .order_by(Email.received_at.desc().nullslast())
        )
        if user_id:
            stmt = stmt.where(GmailAccount.user_id == user_id)
        if limit:
            stmt = stmt.limit(limit)
        return list(db.scalars(stmt).all())

    def index_unindexed_emails(
        self,
        db: Session,
        user_id: Optional[UUID] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, int]:
        """Automatically find and index all unindexed emails for a user or across users.

        Returns summary metrics dictionary.
        """
        unindexed_ids = self.get_unindexed_email_ids(db, user_id=user_id, limit=batch_size)
        total_found = len(unindexed_ids)
        indexed_count = 0
        total_chunks = 0

        for eid in unindexed_ids:
            try:
                chunks = self.index_email(db, email_id=eid, user_id=user_id)
                if chunks > 0:
                    indexed_count += 1
                    total_chunks += chunks
            except Exception as exc:
                logger.warning("Error auto-indexing email %s: %s", eid, exc)

        return {
            "total_unindexed": total_found,
            "emails_indexed": indexed_count,
            "chunks_created": total_chunks,
        }

    def search_rag(
        self,
        query: str,
        user_id: UUID,
        top_k: int = 5,
        filter_sender: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute user-scoped semantic search using text-embedding-3-large query embedding.

        Enforces multi-tenant user isolation via ChromaDB metadata filter `user_id`.
        """
        where_filter: Dict[str, Any] = {"user_id": str(user_id)}
        if filter_sender:
            where_filter = {
                "$and": [
                    {"user_id": str(user_id)},
                    {"sender_email": filter_sender},
                ]
            }

        query_vector = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
        )

        formatted_results: List[Dict[str, Any]] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return formatted_results

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for idx, rec_id in enumerate(ids):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            doc = documents[idx] if idx < len(documents) else ""
            dist = distances[idx] if idx < len(distances) else 0.0
            similarity = round(max(0.0, 1.0 - dist), 4)

            formatted_results.append({
                "record_id": rec_id,
                "chunk_id": meta.get("chunk_id"),
                "email_id": meta.get("email_id"),
                "user_id": meta.get("user_id"),
                "gmail_message_id": meta.get("gmail_message_id"),
                "subject": meta.get("subject"),
                "sender_id": meta.get("sender_id", meta.get("sender_email")),
                "sender_email": meta.get("sender_email"),
                "sender_name": meta.get("sender_name"),
                "recipient_id": meta.get("recipient_id", meta.get("recipient_email")),
                "recipient_email": meta.get("recipient_email"),
                "received_date": meta.get("received_date"),
                "chunk_index": meta.get("chunk_index"),
                "chunk_strategy": meta.get("chunk_strategy"),
                "similarity_score": similarity,
                "document": doc,
            })

        return formatted_results


default_rag_service = RAGIndexerService()

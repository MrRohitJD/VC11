"""PostgreSQL/pgvector retrieval backed by the ``email_embeddings`` table.

The table is the only vector store used by the application.  The metadata
columns are deliberately queried alongside the vector so exact identifiers,
subjects, senders, and timestamps remain reliable even when semantic
similarity is weak.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import sqrt
from typing import Any
from uuid import UUID

from langchain_core.documents import Document
from sqlalchemy import and_, false, func, or_, select

from app.database.connection import SessionLocal
from app.models.email import Email, GmailAccount
from app.models.rag import EmailChunk, EmailEmbedding
from app.rag.mailbox_scope import mailbox_email_condition
from app.rag.embeddings import get_embedding_model
from app.core.config import settings


class PostgresVectorStoreManager:
    collection_name = "postgres_halfvec_3072"

    def __init__(self, embedding_function=None, db_session=None):
        self.embedding_function = embedding_function or get_embedding_model()
        self.db_session = db_session

    def build_filter(self, user_id, sender=None, gmail_message_id=None,
                     start_date=None, end_date=None, start_timestamp=None,
                     end_timestamp=None, custom_filter=None):
        if not user_id:
            raise ValueError("user_id is required")
        clauses = [{"user_id": str(user_id)}]
        if sender:
            clauses.append({"sender_email": sender})
        if gmail_message_id:
            clauses.append({"gmail_message_id": gmail_message_id})
        if start_timestamp is not None:
            clauses.append({"received_timestamp": {"$gte": float(start_timestamp)}})
        elif start_date:
            clauses.append({"received_date": {"$gte": str(start_date)}})
        if end_timestamp is not None:
            clauses.append({"received_timestamp": {"$lte": float(end_timestamp)}})
        elif end_date:
            clauses.append({"received_date": {"$lte": str(end_date)}})
        if custom_filter:
            clauses.append(custom_filter)
        return {"$and": clauses}

    @staticmethod
    def _condition(spec: dict[str, Any]):
        columns = {
            "user_id": GmailAccount.user_id,
            "email_id": Email.id,
            "thread_id": Email.thread_id,
            "gmail_message_id": Email.gmail_message_id,
            "sender_email": EmailEmbedding.sender_email,
            "sender_name": Email.sender_name,
            "email_subject": EmailEmbedding.email_subject,
            "subject": EmailEmbedding.email_subject,
            "embedding_text": EmailEmbedding.embedding_text,
            "embedding_type": EmailEmbedding.embedding_type,
            "vector_collection": EmailEmbedding.vector_collection,
            "email_sent_at": EmailEmbedding.email_sent_at,
            "received_timestamp": func.extract("epoch", Email.received_at),
            "received_date": Email.received_at,
            "chunk_index": EmailChunk.chunk_index,
        }
        conditions = []
        for key, value in spec.items():
            if key == "$and" and isinstance(value, list):
                conditions.append(and_(*(PostgresVectorStoreManager._condition(item) for item in value)))
                continue
            if key == "$or" and isinstance(value, list):
                conditions.append(or_(*(PostgresVectorStoreManager._condition(item) for item in value)))
                continue
            column = columns.get(key)
            if column is None:
                return false()
            if isinstance(value, dict):
                for operator, operand in value.items():
                    if operator == "$in":
                        vals = [UUID(str(x)) if key in {"email_id", "user_id"} else x for x in operand]
                        conditions.append(column.in_(vals) if vals else false())
                    elif operator in {"$gte", "$lte"}:
                        if key in {"received_date", "email_sent_at"}:
                            operand = datetime.fromisoformat(str(operand).replace("Z", "+00:00"))
                            if operand.tzinfo is None:
                                operand = operand.replace(tzinfo=timezone.utc)
                        conditions.append(column >= operand if operator == "$gte" else column <= operand)
                    elif operator in {"$eq", "$ilike", "$contains"}:
                        if operator == "$eq":
                            conditions.append(column == operand)
                        else:
                            conditions.append(column.ilike(f"%{operand}%"))
                    else:
                        return false()
            else:
                if key in {"email_id", "user_id"}:
                    value = UUID(str(value))
                conditions.append(column == value)
        return and_(*conditions)

    def _rows(self, query, user_id, limit, filter_dict=None):
        vector = self.embedding_function.embed_query(query)
        expected_dimensions = settings.embedding_dimensions
        if len(vector) != expected_dimensions:
            raise ValueError(
                f"PostgreSQL vector search requires {expected_dimensions}-dimensional embeddings"
            )
        combined = self.build_filter(user_id, custom_filter=filter_dict)
        filters = (GmailAccount.user_id == UUID(str(user_id)),
                   mailbox_email_condition(),
                   EmailEmbedding.status == "INDEXED", EmailEmbedding.embedding.is_not(None),
                   EmailEmbedding.embedding_dimensions == expected_dimensions,
                   self._condition(combined))
        bind = self.db_session.get_bind() if self.db_session is not None else None
        if bind is None:
            from app.database.connection import engine
            bind = engine

        # SQLite is used by the unit suite and cannot execute pgvector's
        # cosine operator. It still exercises the same table/filter contract
        # with an equivalent in-process ranking fallback.
        if bind.dialect.name == "sqlite":
            stmt = (select(EmailEmbedding, EmailChunk, Email)
                    .join(EmailChunk, EmailEmbedding.chunk_id == EmailChunk.id)
                    .join(Email, EmailChunk.email_id == Email.id)
                    .join(GmailAccount, Email.gmail_account_id == GmailAccount.id)
                    .where(*filters))
            if self.db_session is not None:
                rows = list(self.db_session.execute(stmt).all())
            else:
                with SessionLocal() as db:
                    rows = list(db.execute(stmt).all())

            def cosine_distance(row):
                values = list(row[0].embedding or [])
                numerator = sum(a * b for a, b in zip(values, vector))
                denom = sqrt(sum(a * a for a in values) * sum(b * b for b in vector))
                return 1.0 - (numerator / denom if denom else 0.0)

            rows.sort(key=cosine_distance)
            return [(embedding, chunk, email, cosine_distance((embedding, chunk, email)))
                    for embedding, chunk, email in rows[:limit]]

        distance = EmailEmbedding.embedding.cosine_distance(vector).label("distance")
        stmt = (select(EmailEmbedding, EmailChunk, Email, distance)
                .join(EmailChunk, EmailEmbedding.chunk_id == EmailChunk.id)
                .join(Email, EmailChunk.email_id == Email.id)
                .join(GmailAccount, Email.gmail_account_id == GmailAccount.id)
                .where(*filters).order_by(distance).limit(limit))
        if self.db_session is not None:
            return list(self.db_session.execute(stmt).all())
        with SessionLocal() as db:
            return list(db.execute(stmt).all())

    @staticmethod
    def _document(embedding, chunk, email):
        sent_at = embedding.email_sent_at or email.received_at
        sent_at_text = sent_at.isoformat() if sent_at else ""
        recipients = list(embedding.recipient_emails or [])
        return Document(page_content=chunk.chunk_text, metadata={
            "chunk_id": str(chunk.id), "email_id": str(email.id),
            "gmail_account_id": str(email.gmail_account_id),
            "thread_id": email.thread_id or "",
            "gmail_message_id": email.gmail_message_id or "",
            "sender_email": embedding.sender_email or email.from_email or "", "sender_name": email.sender_name or "",
            "recipient_email": email.to_email or "", "recipient_emails": recipients,
            "subject": embedding.email_subject or email.subject or "", "email_subject": embedding.email_subject or email.subject or "",
            "email_sent_at": sent_at_text, "received_date": sent_at_text,
            "received_at": sent_at_text, "embedding_text": embedding.embedding_text or chunk.chunk_text,
            "embedding_type": embedding.embedding_type or "email",
            "embedding_provider": embedding.embedding_provider or "",
            "embedding_model": embedding.embedding_model or "",
            "embedding_dimensions": embedding.embedding_dimensions,
            "body": chunk.chunk_text,
            "is_read": bool(email.is_read), "is_starred": bool(email.is_starred),
            "has_attachment": bool(email.has_attachment),
            "chunk_index": chunk.chunk_index, "chunk_strategy": chunk.chunk_strategy,
            "source": "postgresql",
        })

    def search_with_scores(self, query, user_id, top_k=5, filter_dict=None):
        return [(self._document(embedding, chunk, email), max(0.0, 1.0 - float(distance)))
                for embedding, chunk, email, distance in self._rows(query, user_id, top_k, filter_dict)]

    def search(self, query, user_id, top_k=5, filter_dict=None):
        return [doc for doc, _ in self.search_with_scores(query, user_id, top_k, filter_dict)]

    def search_mmr_with_scores(self, query, user_id, top_k=5, fetch_k=20,
                               lambda_mult=0.7, filter_dict=None):
        rows = self._rows(query, user_id, max(top_k, fetch_k), filter_dict)
        selected = []
        remaining = list(rows)
        while remaining and len(selected) < top_k:
            def rank(row):
                embedding, _, _, distance = row
                relevance = 1.0 - float(distance)
                vector = list(embedding.embedding)
                def similarity(other):
                    other_vec = list(other[0].embedding)
                    denom = sqrt(sum(x*x for x in vector) * sum(x*x for x in other_vec))
                    return sum(a*b for a, b in zip(vector, other_vec)) / denom if denom else 0.0
                diversity = max((similarity(item) for item in selected), default=0.0)
                return lambda_mult * relevance - (1.0 - lambda_mult) * diversity
            chosen = max(remaining, key=rank)
            selected.append(chosen)
            remaining.remove(chosen)
        return [(self._document(embedding, chunk, email), max(0.0, 1.0 - float(distance)))
                for embedding, chunk, email, distance in selected]

    def search_mmr(self, query, user_id, top_k=5, fetch_k=20, lambda_mult=0.7, filter_dict=None):
        return [doc for doc, _ in self.search_mmr_with_scores(query, user_id, top_k, fetch_k, lambda_mult, filter_dict)]

    def count(self, user_id=None):
        stmt = (select(func.count(EmailEmbedding.id))
                .join(EmailChunk, EmailEmbedding.chunk_id == EmailChunk.id)
                .join(Email, EmailChunk.email_id == Email.id)
                .join(GmailAccount, Email.gmail_account_id == GmailAccount.id)
                .where(EmailEmbedding.status == "INDEXED", EmailEmbedding.embedding.is_not(None)))
        if user_id:
            stmt = stmt.where(GmailAccount.user_id == UUID(str(user_id)))
        with SessionLocal() as db:
            return db.scalar(stmt) or 0

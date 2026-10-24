"""SQLAlchemy ORM models for RAG retrieval, vector embeddings, attachments, and chatbot sessions.

Maintains relational RAG chunks and PostgreSQL pgvector embeddings,
preserving user ownership and relational integrity. PostgreSQL pgvector is the sole vector store.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    ARRAY,
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON
try:
    from pgvector.sqlalchemy import HALFVEC
    EMBEDDING_COLUMN_TYPE = HALFVEC(3072).with_variant(JSON(), "sqlite")
except ImportError:
    HALFVEC = None
    EMBEDDING_COLUMN_TYPE = JSON()

from app.models.email import Base, Email, User, utcnow


class EmailChunk(Base):
    """Semantic chunk of an email message stored for RAG search and retrieval."""

    __tablename__ = "email_chunks"
    __table_args__ = (
        UniqueConstraint("email_id", "chunk_index", name="uq_email_chunks_email_index"),
        Index("ix_email_chunks_email_id", "email_id"),
        Index("ix_email_chunks_content_hash", "content_hash"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    email_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("emails.id", ondelete="CASCADE"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    start_index: Mapped[int | None] = mapped_column(Integer)
    end_index: Mapped[int | None] = mapped_column(Integer)
    chunk_strategy: Mapped[str] = mapped_column(String(50), default="recursive_character", server_default="recursive_character", nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), default="", server_default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )

    email: Mapped[Email] = relationship(back_populates="chunks")
    embeddings: Mapped[list["EmailEmbedding"]] = relationship(back_populates="chunk", cascade="all, delete-orphan")
    chat_retrievals: Mapped[list["ChatRetrieval"]] = relationship(back_populates="chunk", cascade="all, delete-orphan")


class EmailEmbedding(Base):
    """Dense pgvector embedding and searchable email metadata for one chunk."""

    __tablename__ = "email_embeddings"
    __table_args__ = (
        UniqueConstraint("chroma_collection", "chroma_record_id", name="uq_email_embeddings_collection_record"),
        UniqueConstraint("chunk_id", "chroma_collection", name="uq_email_embeddings_chunk_collection"),
        CheckConstraint("status IN ('PENDING', 'INDEXED', 'FAILED', 'OUTDATED')", name="ck_email_embeddings_status"),
        Index("ix_email_embeddings_chunk_id", "chunk_id"),
        Index("ix_email_embeddings_status", "status"),
        Index("ix_email_embeddings_content_hash", "content_hash"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    chunk_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("email_chunks.id", ondelete="CASCADE"), nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(EMBEDDING_COLUMN_TYPE)
    email_id: Mapped[UUID | None] = mapped_column(Uuid, ForeignKey("emails.id", ondelete="CASCADE"), nullable=True)
    email_subject: Mapped[str | None] = mapped_column(Text)
    sender_email: Mapped[str | None] = mapped_column(Text)
    recipient_emails: Mapped[list[str] | None] = mapped_column(ARRAY(Text).with_variant(JSON(), "sqlite"))
    email_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    embedding_text: Mapped[str | None] = mapped_column(Text)
    embedding_type: Mapped[str | None] = mapped_column(String(30), default="email", server_default="email")
    chroma_collection: Mapped[str] = mapped_column(
        String(100), default="postgres_halfvec_3072", server_default="postgres_halfvec_3072", nullable=False
    )
    chroma_record_id: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_provider: Mapped[str] = mapped_column(
        String(50), default="azure_openai", server_default="azure_openai", nullable=False
    )
    embedding_model: Mapped[str] = mapped_column(
        String(100), default="text-embedding-3-large", server_default="text-embedding-3-large", nullable=False
    )
    embedding_dimensions: Mapped[int] = mapped_column(Integer, default=3072, server_default="3072", nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="INDEXED", server_default="INDEXED", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )

    chunk: Mapped[EmailChunk] = relationship(back_populates="embeddings")


class Attachment(Base):
    """Metadata and extracted content for Gmail attachments."""

    __tablename__ = "attachments"
    __table_args__ = (
        UniqueConstraint("email_id", "gmail_attachment_id", name="uq_attachments_email_attachment_id"),
        CheckConstraint(
            "extraction_status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'SKIPPED')",
            name="ck_attachments_extraction_status",
        ),
        Index("ix_attachments_email_id", "email_id"),
        Index("ix_attachments_extraction_status", "extraction_status"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    email_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("emails.id", ondelete="CASCADE"), nullable=False)
    gmail_attachment_id: Mapped[str | None] = mapped_column(String(255))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(150))
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    storage_uri: Mapped[str | None] = mapped_column(Text)
    extracted_text: Mapped[str | None] = mapped_column(Text)
    extraction_status: Mapped[str] = mapped_column(
        String(30), default="PENDING", server_default="PENDING", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )

    email: Mapped[Email] = relationship(back_populates="attachments")


class ChatSession(Base):
    """RAG conversational session linked to an authenticated user."""

    __tablename__ = "chat_sessions"
    __table_args__ = (
        Index("ix_chat_sessions_user_id", "user_id"),
        Index("ix_chat_sessions_updated_at", "updated_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )

    user: Mapped[User] = relationship(back_populates="chat_sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    retrievals: Mapped[list["ChatRetrieval"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """User prompts and AI responses within a conversational chat session."""

    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="ck_chat_messages_role"),
        Index("ix_chat_messages_session_created", "chat_session_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    chat_session_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")
    retrievals: Mapped[list["ChatRetrieval"]] = relationship(back_populates="message", cascade="all, delete-orphan")


class ChatRetrieval(Base):
    """Audit log of email chunks retrieved to ground each AI chat message response."""

    __tablename__ = "chat_retrievals"
    __table_args__ = (
        UniqueConstraint("chat_message_id", "chunk_id", name="uq_chat_retrievals_message_chunk"),
        Index("ix_chat_retrievals_message_id", "chat_message_id"),
        Index("ix_chat_retrievals_chunk_id", "chunk_id"),
        Index("ix_chat_retrievals_session_id", "chat_session_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    chat_session_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    chat_message_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False)
    chunk_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("email_chunks.id", ondelete="CASCADE"), nullable=False)
    similarity_score: Mapped[float | None] = mapped_column(Float)
    retrieval_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )

    session: Mapped[ChatSession] = relationship(back_populates="retrievals")
    message: Mapped[ChatMessage] = relationship(back_populates="retrievals")
    chunk: Mapped[EmailChunk] = relationship(back_populates="chat_retrievals")


class EmailSendApproval(Base):
    """Optional approval queue for AI-assisted drafted outbound emails."""

    __tablename__ = "email_send_approvals"
    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'APPROVED', 'REJECTED', 'SENT', 'CANCELLED')",
            name="ck_email_send_approvals_status",
        ),
        Index("ix_email_send_approvals_user_status", "user_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    recipient_email: Mapped[str] = mapped_column(String(320), nullable=False)
    subject: Mapped[str | None] = mapped_column(Text)
    draft_body: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", server_default="PENDING", nullable=False)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )

    user: Mapped[User] = relationship(back_populates="send_approvals")

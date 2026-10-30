"""Central ORM mappings for the Gmail AI Dashboard persistence boundary.

Python attribute names intentionally preserve the API/service vocabulary while
the physical column names are explicit. This confines future database naming
changes to this module and a versioned migration.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, Float, ForeignKey, ForeignKeyConstraint,
    Index, Integer, String, Text, UniqueConstraint, Uuid, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, synonym
from sqlalchemy.types import JSON


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


JSONType = JSON().with_variant(JSONB, "postgresql")


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    google_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    picture: Mapped[str | None] = mapped_column(String(500))
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"), nullable=False)
    developer_database_url: Mapped[str | None] = mapped_column(Text)
    developer_llm_provider: Mapped[str | None] = mapped_column(String(20))
    developer_llm_api_key: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    gmail_accounts: Mapped[list["GmailAccount"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    chat_sessions: Mapped[list["ChatSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    send_approvals: Mapped[list["EmailSendApproval"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class GmailAccount(Base):
    __tablename__ = "gmail_accounts"
    __table_args__ = (
        UniqueConstraint("user_id", "email_address", name="uq_gmail_accounts_user_email"),
        UniqueConstraint("id", "user_id", name="uq_gmail_accounts_id_user"),
        Index("ix_gmail_accounts_user_id", "user_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    # The public/service attribute remains ``email``; the physical column is clearer.
    email: Mapped[str] = mapped_column("email_address", String(320), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), default="gmail", server_default="gmail", nullable=False)
    gmail_user_id: Mapped[str | None] = mapped_column(String(255))
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(Text)
    token_expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    scope: Mapped[str | None] = mapped_column(Text)
    sync_status: Mapped[str] = mapped_column(String(20), default="IDLE", server_default="IDLE", nullable=False)
    inbox_sync_page_token: Mapped[str | None] = mapped_column(Text)
    inbox_sync_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    history_id: Mapped[str | None] = mapped_column(String(255))
    ai_classification_enabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    user: Mapped[User] = relationship(back_populates="gmail_accounts")
    emails: Mapped[list["Email"]] = relationship(back_populates="gmail_account", cascade="all, delete-orphan")
    labels: Mapped[list["Label"]] = relationship(back_populates="gmail_account", cascade="all, delete-orphan")
    sync_history: Mapped[list["SyncHistory"]] = relationship(back_populates="gmail_account", cascade="all, delete-orphan")
    classification_jobs: Mapped[list["LlmClassificationJob"]] = relationship(back_populates="gmail_account", cascade="all, delete-orphan")


class Email(Base):
    __tablename__ = "emails"
    __table_args__ = (
        UniqueConstraint("gmail_account_id", "gmail_message_id", name="uq_emails_account_message"),
        UniqueConstraint("id", "gmail_account_id", name="uq_emails_id_account"),
        Index("ix_emails_account_received", "gmail_account_id", "received_at"),
        Index("ix_emails_account_thread", "gmail_account_id", "gmail_thread_id"),
        Index("ix_emails_account_unread", "gmail_account_id", "is_read"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    gmail_account_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("gmail_accounts.id", ondelete="CASCADE"), nullable=False)
    gmail_message_id: Mapped[str] = mapped_column(String(255), nullable=False)
    thread_id: Mapped[str | None] = mapped_column("gmail_thread_id", String(255))
    from_email: Mapped[str | None] = mapped_column("sender_email", Text)
    sender_name: Mapped[str | None] = mapped_column(String(255))
    to_email: Mapped[str | None] = mapped_column("recipient_metadata", Text)
    cc: Mapped[str | None] = mapped_column("cc_metadata", Text)
    bcc: Mapped[str | None] = mapped_column("bcc_metadata", Text)
    subject: Mapped[str | None] = mapped_column(Text)
    snippet: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column("body_text", Text)
    body_html: Mapped[str | None] = mapped_column(Text)
    received_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"), nullable=False)
    is_starred: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"), nullable=False)
    has_attachment: Mapped[bool] = mapped_column("has_attachments", Boolean, default=False, server_default=text("false"), nullable=False)
    attachments_meta: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(JSONType)
    size_estimate: Mapped[int | None] = mapped_column(Integer)
    history_id: Mapped[str | None] = mapped_column(String(255))
    raw_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    gmail_account: Mapped[GmailAccount] = relationship(back_populates="emails")
    labels: Mapped[list["EmailLabel"]] = relationship(back_populates="email", cascade="all, delete-orphan", overlaps="emails,label")
    classifications: Mapped[list["EmailClassification"]] = relationship(back_populates="email", cascade="all, delete-orphan")
    classification_jobs: Mapped[list["LlmClassificationJob"]] = relationship(back_populates="email")
    chunks: Mapped[list["EmailChunk"]] = relationship(back_populates="email", cascade="all, delete-orphan")
    attachments: Mapped[list["Attachment"]] = relationship(back_populates="email", cascade="all, delete-orphan")


class Label(Base):
    __tablename__ = "labels"
    __table_args__ = (
        UniqueConstraint("gmail_account_id", "gmail_label_id", name="uq_labels_account_gmail_label"),
        UniqueConstraint("id", "gmail_account_id", name="uq_labels_id_account"),
        Index("ix_labels_account_name", "gmail_account_id", "name"),
        Index("ix_labels_gmail_label_account", "gmail_label_id", "gmail_account_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    gmail_account_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("gmail_accounts.id", ondelete="CASCADE"), nullable=False)
    label_id: Mapped[str] = mapped_column("gmail_label_id", String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str | None] = mapped_column("label_type", String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    gmail_account: Mapped[GmailAccount] = relationship(back_populates="labels")
    emails: Mapped[list["EmailLabel"]] = relationship(back_populates="label", cascade="all, delete-orphan", overlaps="labels,email")


class EmailLabel(Base):
    __tablename__ = "email_labels"
    __table_args__ = (
        ForeignKeyConstraint(["email_id", "gmail_account_id"], ["emails.id", "emails.gmail_account_id"], ondelete="CASCADE", name="fk_email_labels_email_account"),
        ForeignKeyConstraint(["label_id", "gmail_account_id"], ["labels.id", "labels.gmail_account_id"], ondelete="CASCADE", name="fk_email_labels_label_account"),
        Index("ix_email_labels_label_id", "label_id"),
        Index("ix_email_labels_account_label_email", "gmail_account_id", "label_id", "email_id"),
    )

    email_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    label_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    gmail_account_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    email: Mapped[Email] = relationship(back_populates="labels", overlaps="emails,label")
    label: Mapped[Label] = relationship(back_populates="emails", overlaps="labels,email")


class AiCategory(Base):
    __tablename__ = "ai_categories"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    classifications: Mapped[list["EmailClassification"]] = relationship(back_populates="category")


class EmailClassification(Base):
    __tablename__ = "email_classifications"
    __table_args__ = (
        CheckConstraint("confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)", name="ck_email_classifications_confidence"),
        CheckConstraint("classification_status IN ('PENDING', 'SUCCEEDED', 'FAILED')", name="ck_email_classifications_status"),
        Index("ix_email_classifications_email_current", "email_id", "is_current"),
        # Dashboard metrics and charts only need the current successful result.
        # Keeping this index partial avoids indexing historical, pending, and
        # failed classifications while making the email_id join selective.
        Index(
            "ix_email_classifications_dashboard_lookup",
            "email_id",
            "category_id",
            postgresql_where=text("is_current = true AND classification_status = 'SUCCEEDED'"),
            sqlite_where=text("is_current = 1 AND classification_status = 'SUCCEEDED'"),
        ),
        Index("ix_email_classifications_category_id", "category_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    email_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("emails.id", ondelete="CASCADE"), nullable=False)
    category_id: Mapped[UUID | None] = mapped_column(Uuid, ForeignKey("ai_categories.id", ondelete="RESTRICT"))
    confidence_score: Mapped[float | None] = mapped_column(Float)
    model_name: Mapped[str | None] = mapped_column(String(150))
    model_version: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(100))
    classification_status: Mapped[str] = mapped_column(String(20), default="SUCCEEDED", server_default="SUCCEEDED", nullable=False)
    classification_reason: Mapped[str | None] = mapped_column(Text)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"), nullable=False)
    classified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    # Transitional Python aliases; physical schema and new code use the names above.
    label_id = synonym("category_id")
    confidence = synonym("confidence_score")
    reason = synonym("classification_reason")

    @property
    def label(self) -> AiCategory | None:
        """Deprecated compatibility alias for callers that used LLM labels."""
        return self.category

    email: Mapped[Email] = relationship(back_populates="classifications")
    category: Mapped[AiCategory | None] = relationship(back_populates="classifications")


class SyncHistory(Base):
    __tablename__ = "sync_history"
    __table_args__ = (
        CheckConstraint("status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED')", name="ck_sync_history_status"),
        Index("ix_sync_history_account_started", "gmail_account_id", "started_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    gmail_account_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("gmail_accounts.id", ondelete="CASCADE"), nullable=False)
    trigger: Mapped[str] = mapped_column("sync_type", String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    emails_found: Mapped[int] = mapped_column("messages_fetched", Integer, default=0, server_default="0", nullable=False)
    emails_synced: Mapped[int] = mapped_column("messages_inserted", Integer, default=0, server_default="0", nullable=False)
    messages_updated: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    labels_synced: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    error: Mapped[str | None] = mapped_column("error_message", Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    gmail_account: Mapped[GmailAccount] = relationship(back_populates="sync_history")


# Register dependent job and RAG mappers for consumers that import this module
# directly (for example migration and model-only test tooling).
import app.models.llm  # noqa: E402,F401
import app.models.rag  # noqa: E402,F401

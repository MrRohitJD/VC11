"""Durable AI job mapping and compatibility exports.

Categories and classifications live in the normalized core models. The old
class names remain imports-only aliases so callers can be migrated gradually.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.email import AiCategory, Base, Email, EmailClassification, GmailAccount, utcnow


class LlmClassificationJob(Base):
    __tablename__ = "llm_classification_jobs"
    __table_args__ = (
        CheckConstraint("status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'PARTIAL')", name="ck_llm_jobs_status"),
        CheckConstraint("attempts >= 0 AND max_attempts > 0 AND attempts <= max_attempts", name="ck_llm_jobs_attempts"),
        Index("ix_llm_jobs_claim", "status", "scheduled_at"),
        Index("ix_llm_jobs_account_email", "gmail_account_id", "email_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    gmail_account_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("gmail_accounts.id", ondelete="CASCADE"), nullable=False)
    email_id: Mapped[UUID | None] = mapped_column(Uuid, ForeignKey("emails.id", ondelete="CASCADE"))
    job_type: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", server_default="PENDING", nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, server_default="3", nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Preserved UI metrics; they do not duplicate per-email classification data.
    emails_discovered: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    emails_classified: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    emails_remaining: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    emails_failed: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    gmail_account: Mapped[GmailAccount] = relationship(back_populates="classification_jobs")
    email: Mapped[Email | None] = relationship(back_populates="classification_jobs")

    @property
    def last_error(self) -> str | None:
        return self.error_message

    @last_error.setter
    def last_error(self, value: str | None) -> None:
        self.error_message = value


LlmLabel = AiCategory
LlmEmailClassification = EmailClassification

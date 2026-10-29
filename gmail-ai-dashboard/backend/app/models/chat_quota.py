"""Durable daily assistant request counters."""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.email import Base


class ChatQuota(Base):
    __tablename__ = "chat_quotas"

    identity: Mapped[str] = mapped_column(String(80), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, nullable=False)


class ChatConcurrencyLock(Base):
    """A per-user mutex used to reserve quotas safely across workers."""

    __tablename__ = "chat_concurrency_locks"

    identity: Mapped[str] = mapped_column(String(80), primary_key=True)


class ChatRequestLeaseRecord(Base):
    """An in-flight chat request, expiring automatically after worker failure."""

    __tablename__ = "chat_request_leases"
    __table_args__ = (
        Index("ix_chat_request_leases_identity_expires_at", "identity", "expires_at"),
        Index("ix_chat_request_leases_expires_at", "expires_at"),
    )

    lease_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    identity: Mapped[str] = mapped_column(String(80), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

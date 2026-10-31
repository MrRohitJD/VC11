"""Models for multi-layer rate limiting, visitor tracking, security events, and AI accounting."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.email import Base


class RateLimitBucket(Base):
    """Atomic fixed-window counter buckets for multi-window rate limiting."""

    __tablename__ = "rate_limit_buckets"
    __table_args__ = (
        UniqueConstraint("bucket_key", "window_seconds", "window_start", name="uq_rate_limit_buckets_key_window"),
        Index("ix_rate_limit_buckets_expires_at", "expires_at"),
        Index("ix_rate_limit_buckets_key", "bucket_key"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    bucket_key: Mapped[str] = mapped_column(String(128), nullable=False)
    window_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DemoVisitor(Base):
    """Persistent tracking and abuse evaluation for anonymous demo visitors."""

    __tablename__ = "demo_visitors"
    __table_args__ = (
        Index("ix_demo_visitors_visitor_hash", "visitor_hash"),
        Index("ix_demo_visitors_blocked_until", "blocked_until"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    visitor_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_blocked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    blocked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    abuse_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ip_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(255), nullable=True)


class SecurityEvent(Base):
    """Audit log for abuse detection, rate limit hits, and authentication anomalies."""

    __tablename__ = "security_events"
    __table_args__ = (
        Index("ix_security_events_created_at", "created_at"),
        Index("ix_security_events_ip_hash", "ip_hash"),
        Index("ix_security_events_visitor_hash", "visitor_hash"),
        Index("ix_security_events_event_type", "event_type"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    visitor_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ip_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[UUID | None] = mapped_column(nullable=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, default="INFO")
    endpoint: Mapped[str] = mapped_column(String(256), nullable=False)
    http_method: Mapped[str] = mapped_column(String(10), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    detail: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AiUsageLog(Base):
    """Granular accounting of AI tokens and estimated financial costs."""

    __tablename__ = "ai_usage_logs"
    __table_args__ = (
        Index("ix_ai_usage_logs_created_at", "created_at"),
        Index("ix_ai_usage_logs_user_id", "user_id"),
        Index("ix_ai_usage_logs_visitor_hash", "visitor_hash"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID | None] = mapped_column(nullable=True)
    visitor_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ip_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    operation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

"""Atomic PostgreSQL rate-limiting engine with multi-window sliding/fixed bucket accounting."""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from uuid import UUID

from sqlalchemy import delete, text
from sqlalchemy.orm import Session

from app.models.rate_limit import RateLimitBucket
from app.services.security_events import record_security_event

logger = logging.getLogger(__name__)


def check_rate_limit(
    db: Session,
    bucket_key: str,
    limit: int,
    window_seconds: int,
    ip_hash: str = "",
    visitor_hash: Optional[str] = None,
    user_id: Optional[UUID] = None,
    endpoint: str = "",
    http_method: str = "",
) -> Tuple[bool, int, int, int]:
    """Atomically check and increment a rate limit counter.

    Uses PostgreSQL atomic INSERT ... ON CONFLICT DO UPDATE RETURNING.
    Guarantees no race conditions between checking and incrementing counters across multiple dynos.

    Returns:
        (allowed, retry_after_seconds, current_count, limit)
    """
    now = datetime.now(timezone.utc)
    now_epoch = int(now.timestamp())
    window_start_epoch = (now_epoch // window_seconds) * window_seconds
    window_start = datetime.fromtimestamp(window_start_epoch, tz=timezone.utc)
    # Store expiration with safety margin for cleanup
    expires_at = window_start + timedelta(seconds=window_seconds * 2 + 60)

    # Dialect-safe atomic query
    # Both PostgreSQL and SQLite (>= 3.35) support INSERT ... ON CONFLICT DO UPDATE ... RETURNING
    stmt = text("""
        INSERT INTO rate_limit_buckets (bucket_key, window_seconds, window_start, request_count, expires_at, created_at, updated_at)
        VALUES (:key, :win, :start, 1, :exp, :now, :now)
        ON CONFLICT (bucket_key, window_seconds, window_start)
        DO UPDATE SET
            request_count = rate_limit_buckets.request_count + 1,
            updated_at = :now
        RETURNING request_count;
    """)

    try:
        res = db.execute(stmt, {
            "key": bucket_key[:128],
            "win": window_seconds,
            "start": window_start,
            "exp": expires_at,
            "now": now,
        })
        current_count = res.scalar_one()
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("Atomic rate limit query error for key %s: %s", bucket_key, exc)
        # Fail open rather than crash user requests if DB has a transient issue,
        # but log warning
        return True, 0, 1, limit

    if current_count > limit:
        window_end_epoch = window_start_epoch + window_seconds
        retry_after = max(1, window_end_epoch - now_epoch)

        if ip_hash:
            record_security_event(
                db=db,
                ip_hash=ip_hash,
                visitor_hash=visitor_hash,
                user_id=user_id,
                event_type="RATE_LIMIT_EXCEEDED",
                severity="WARNING" if current_count > limit * 2 else "INFO",
                endpoint=endpoint,
                http_method=http_method,
                status_code=429,
                detail=f"Rate limit exceeded on bucket '{bucket_key}': {current_count}/{limit} in {window_seconds}s",
                metadata={"bucket_key": bucket_key, "count": current_count, "limit": limit, "window_seconds": window_seconds},
            )
        return False, retry_after, current_count, limit

    return True, 0, current_count, limit


def cleanup_expired_buckets(db: Session) -> int:
    """Purge stale rate limit buckets to keep the table compact."""
    now = datetime.now(timezone.utc)
    try:
        deleted = db.execute(
            delete(RateLimitBucket).where(RateLimitBucket.expires_at <= now)
        ).rowcount
        db.commit()
        return deleted
    except Exception as exc:
        db.rollback()
        logger.warning("Could not clean up expired rate limit buckets: %s", exc)
        return 0

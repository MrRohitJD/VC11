"""Gmail API synchronization limits, outbound email quotas, duplicate send prevention, and idempotency."""

import hashlib
import hmac
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.email import User
from app.services.rate_limiter import check_rate_limit
from app.services.security_events import record_security_event

logger = logging.getLogger(__name__)

# Outbound sending limits
MAX_EMAILS_PER_HOUR = 10
MAX_EMAILS_PER_DAY = 30
MAX_RECIPIENTS_TOTAL = 10
DUPLICATE_WINDOW_SECONDS = 60

# In-memory duplicate send cache: {fingerprint: timestamp}
_RECENT_SENDS: dict[str, float] = {}
_SEND_LOCK = threading.Lock()

# Idempotency cache: {idempotency_key: (result, timestamp)}
_IDEMPOTENCY_CACHE: dict[str, Tuple[Dict[str, Any], float]] = {}
_IDEMPOTENCY_LOCK = threading.Lock()
IDEMPOTENCY_TTL_SECONDS = 300  # 5 minutes


def _clean_caches(now: float) -> None:
    """Purge stale duplicate fingerprints and idempotency records."""
    expired_sends = [k for k, exp in _RECENT_SENDS.items() if exp <= now]
    for k in expired_sends:
        del _RECENT_SENDS[k]

    expired_idemp = [k for k, (_, exp) in _IDEMPOTENCY_CACHE.items() if exp <= now]
    for k in expired_idemp:
        del _IDEMPOTENCY_CACHE[k]


def check_sync_rate_limit(db: Session, user: User, ip_hash: str) -> None:
    """Enforce manual Gmail synchronization frequency limit (max 3/hour per user)."""
    if user.is_demo:
        raise HTTPException(status_code=403, detail="Gmail sync is unavailable in demo mode")

    bucket_key = f"gmail:sync:{user.id}"
    allowed, retry_after, count, limit = check_rate_limit(
        db=db,
        bucket_key=bucket_key,
        limit=3,
        window_seconds=3600,
        ip_hash=ip_hash,
        user_id=user.id,
        endpoint="/api/gmail/sync",
        http_method="POST",
    )

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Gmail sync rate limit reached ({limit} syncs per hour). Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


def check_outbound_email_quota(
    db: Session,
    user: User,
    recipient: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    ip_hash: str = "",
) -> Optional[Dict[str, Any]]:
    """Enforce outbound email rate limits, duplicate detection, and idempotency.

    Returns:
        Cached result if idempotency key matches a prior successful delivery, else None.
    """
    now = time.time()
    with _SEND_LOCK:
        _clean_caches(now)

    # 1. Check idempotency cache if key provided
    if idempotency_key:
        with _IDEMPOTENCY_LOCK:
            cached = _IDEMPOTENCY_CACHE.get(idempotency_key)
            if cached and cached[1] > now:
                logger.info("Returning idempotent response for key %s", idempotency_key)
                return cached[0]

    # 2. In demo mode, sending is simulated; skip real quota checks
    if user.is_demo:
        return None

    # 3. Validate recipient count limit
    total_recipients = 1
    if cc:
        total_recipients += len([c for c in cc.split(",") if c.strip()])
    if bcc:
        total_recipients += len([b for b in bcc.split(",") if b.strip()])

    if total_recipients > MAX_RECIPIENTS_TOTAL:
        raise HTTPException(
            status_code=400,
            detail=f"Total recipients ({total_recipients}) exceeds maximum allowed of {MAX_RECIPIENTS_TOTAL}.",
        )

    # 4. Check hourly outbound limit
    hourly_key = f"gmail:send:hour:{user.id}"
    allowed_h, retry_h, _, _ = check_rate_limit(
        db=db,
        bucket_key=hourly_key,
        limit=MAX_EMAILS_PER_HOUR,
        window_seconds=3600,
        ip_hash=ip_hash,
        user_id=user.id,
        endpoint="/api/emails/send",
        http_method="POST",
    )
    if not allowed_h:
        raise HTTPException(
            status_code=429,
            detail=f"Outbound email hourly limit reached ({MAX_EMAILS_PER_HOUR}/hour). Try again in {retry_h} seconds.",
            headers={"Retry-After": str(retry_h)},
        )

    # 5. Check daily outbound limit
    daily_key = f"gmail:send:day:{user.id}"
    allowed_d, retry_d, _, _ = check_rate_limit(
        db=db,
        bucket_key=daily_key,
        limit=MAX_EMAILS_PER_DAY,
        window_seconds=86400,
        ip_hash=ip_hash,
        user_id=user.id,
        endpoint="/api/emails/send",
        http_method="POST",
    )
    if not allowed_d:
        raise HTTPException(
            status_code=429,
            detail=f"Outbound email daily limit reached ({MAX_EMAILS_PER_DAY}/day). Try again tomorrow.",
            headers={"Retry-After": str(retry_d)},
        )

    # 6. Duplicate-send detection within 60 seconds
    fingerprint = hashlib.sha256(
        f"{user.id}:{recipient.strip().lower()}:{subject.strip()}:{body.strip()}".encode()
    ).hexdigest()

    with _SEND_LOCK:
        last_sent = _RECENT_SENDS.get(fingerprint)
        if last_sent and (now - last_sent) < DUPLICATE_WINDOW_SECONDS:
            remaining = int(DUPLICATE_WINDOW_SECONDS - (now - last_sent))
            record_security_event(
                db=db,
                ip_hash=ip_hash,
                user_id=user.id,
                event_type="DUPLICATE_EMAIL_SEND_BLOCKED",
                severity="WARNING",
                endpoint="/api/emails/send",
                http_method="POST",
                status_code=409,
                detail=f"Duplicate email send to {recipient} blocked within {remaining}s",
            )
            raise HTTPException(
                status_code=409,
                detail=f"An identical email was sent recently. Please wait {remaining} seconds before resending.",
            )
        _RECENT_SENDS[fingerprint] = now + DUPLICATE_WINDOW_SECONDS

    return None


def record_successful_send(idempotency_key: Optional[str], result: Dict[str, Any]) -> None:
    """Cache the delivery result if an idempotency key was supplied."""
    if idempotency_key:
        now = time.time()
        with _IDEMPOTENCY_LOCK:
            _IDEMPOTENCY_CACHE[idempotency_key] = (result, now + IDEMPOTENCY_TTL_SECONDS)

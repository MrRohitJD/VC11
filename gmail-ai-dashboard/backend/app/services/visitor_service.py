"""Cryptographic visitor identity tracking, abuse scoring, and demo visitor lifecycle."""

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from uuid import uuid4

from fastapi import HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.rate_limit import DemoVisitor
from app.services.ip_trust import digest_ip, extract_client_ip
from app.services.security_events import record_security_event

logger = logging.getLogger(__name__)

ABUSE_THRESHOLD = 50
BLOCK_DURATION_MINUTES = 15


def _digest_visitor(visitor_id: str) -> str:
    """HMAC-SHA256 signature for visitor token."""
    return hmac.new(settings.session_secret.encode(), f"visitor:{visitor_id}".encode(), hashlib.sha256).hexdigest()


def visitor_hash(visitor_id: str) -> str:
    """Opaque HMAC hash stored in database and bucket keys."""
    return hmac.new(settings.session_secret.encode(), f"visitor_hash:{visitor_id}".encode(), hashlib.sha256).hexdigest()[:32]


def validate_visitor_cookie(raw_cookie: str) -> Tuple[Optional[str], bool]:
    """Validate visitor cookie structure and cryptographic signature.

    Returns:
        (visitor_id, is_tampered)
    """
    if not raw_cookie:
        return None, False

    visitor_id, separator, signature = raw_cookie.partition(".")
    if not separator or not visitor_id or len(visitor_id) > 128:
        return None, True

    expected = _digest_visitor(visitor_id)
    if not hmac.compare_digest(signature, expected):
        return None, True

    return visitor_id, False


def get_visitor_id(request: Request) -> Tuple[Optional[str], bool]:
    """Extract and validate visitor identifier from incoming request cookies."""
    raw_cookie = request.cookies.get(settings.demo_visitor_cookie_name, "")
    return validate_visitor_cookie(raw_cookie)


def set_visitor_cookie(response: Response, visitor_id: Optional[str] = None) -> str:
    """Issue a cryptographically signed visitor cookie to the response."""
    vid = visitor_id or secrets.token_urlsafe(32)
    signed_value = f"{vid}.{_digest_visitor(vid)}"
    response.set_cookie(
        key=settings.demo_visitor_cookie_name,
        value=signed_value,
        httponly=True,
        secure=settings.secure_cookies,
        samesite="lax",
        path="/",
        max_age=settings.demo_visitor_ttl_days * 86400,
    )
    return vid


def get_or_register_visitor(
    db: Session,
    request: Request,
    response: Optional[Response] = None,
) -> Tuple[str, str, DemoVisitor]:
    """Retrieve existing visitor or register new visitor in database.

    Returns:
        (visitor_id, v_hash, visitor_record)
    """
    client_ip = extract_client_ip(request)
    ip_h = digest_ip(client_ip)
    ua = (request.headers.get("user-agent") or "")[:255]
    now = datetime.now(timezone.utc)

    raw_cookie = request.cookies.get(settings.demo_visitor_cookie_name, "")
    vid, is_tampered = validate_visitor_cookie(raw_cookie)

    if is_tampered:
        record_security_event(
            db=db,
            ip_hash=ip_h,
            event_type="COOKIE_TAMPERING",
            severity="WARNING",
            endpoint=request.url.path if hasattr(request, "scope") and "path" in request.scope else "/api",
            http_method=request.method,
            status_code=401,
            detail="Tampered visitor cookie detected and rejected",
        )
        vid = None

    if not vid:
        vid = secrets.token_urlsafe(32)
        if response is not None:
            set_visitor_cookie(response, vid)

    v_hash = visitor_hash(vid)

    # Lookup or create visitor record
    visitor = db.scalar(select(DemoVisitor).where(DemoVisitor.visitor_hash == v_hash))
    if visitor is None:
        visitor = DemoVisitor(
            id=uuid4(),
            visitor_hash=v_hash,
            first_seen_at=now,
            last_seen_at=now,
            is_blocked=False,
            blocked_until=None,
            abuse_score=0,
            total_requests=1,
            ip_hash=ip_h,
            user_agent=ua,
        )
        db.add(visitor)
    else:
        visitor.last_seen_at = now
        visitor.total_requests += 1
        visitor.ip_hash = ip_h
        visitor.user_agent = ua

    try:
        db.commit()
    except Exception:
        db.rollback()

    # Check if visitor is blocked
    if visitor.is_blocked:
        raise HTTPException(
            status_code=403,
            detail="Your visitor session has been blocked due to abuse. Please contact administrator.",
        )

    b_until = visitor.blocked_until
    if b_until is not None and b_until.tzinfo is None:
        b_until = b_until.replace(tzinfo=timezone.utc)

    if b_until and b_until > now:
        remaining = int((b_until - now).total_seconds())
        raise HTTPException(
            status_code=429,
            detail=f"Your visitor session is in temporary cooldown. Try again in {remaining} seconds.",
            headers={"Retry-After": str(remaining)},
        )

    return vid, v_hash, visitor


def increment_abuse_score(
    db: Session,
    visitor_hash_val: str,
    ip_hash_val: str,
    points: int,
    reason: str,
    endpoint: str = "",
) -> None:
    """Increase abuse score for a visitor and trigger temporary block if threshold reached."""
    now = datetime.now(timezone.utc)
    try:
        visitor = db.scalar(select(DemoVisitor).where(DemoVisitor.visitor_hash == visitor_hash_val))
        if visitor is not None:
            visitor.abuse_score += points
            if visitor.abuse_score >= ABUSE_THRESHOLD and not visitor.blocked_until:
                visitor.blocked_until = now + timedelta(minutes=BLOCK_DURATION_MINUTES)
                record_security_event(
                    db=db,
                    ip_hash=ip_hash_val,
                    visitor_hash=visitor_hash_val,
                    event_type="TEMPORARY_BLOCK_ISSUED",
                    severity="WARNING",
                    endpoint=endpoint,
                    http_method="",
                    status_code=429,
                    detail=f"Visitor blocked for {BLOCK_DURATION_MINUTES}m due to abuse score {visitor.abuse_score}: {reason}",
                )
            db.commit()
    except Exception as exc:
        logger.warning("Could not increment abuse score: %s", type(exc).__name__)
        try:
            db.rollback()
        except Exception:
            pass

"""Reserve one of ten daily assistant requests before running the agent."""

import hashlib
import hmac
import secrets
from datetime import datetime, timezone

from fastapi import HTTPException, Request, Response
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.chat_quota import ChatQuota
from app.models.email import User

DAILY_LIMIT = 5
DEMO_VISITOR_COOKIE = "gmail_ai_demo_visitor"


def _digest(value: str) -> str:
    return hmac.new(settings.session_secret.encode(), value.encode(), hashlib.sha256).hexdigest()


def _visitor_id(request: Request) -> str | None:
    raw_cookie = request.cookies.get(DEMO_VISITOR_COOKIE, "")
    visitor, separator, signature = raw_cookie.partition(".")
    if separator and visitor and len(visitor) <= 128 and hmac.compare_digest(signature, _digest(visitor)):
        return visitor
    return None


def set_demo_visitor_cookie(request: Request, response: Response) -> None:
    """Keep a signed visitor identifier across demo sessions and logouts."""
    visitor = _visitor_id(request) or secrets.token_urlsafe(32)
    response.set_cookie(
        DEMO_VISITOR_COOKIE, f"{visitor}.{_digest(visitor)}",
        httponly=True, secure=settings.secure_cookies, samesite="lax",
        max_age=365 * 86400,
    )


def _identities(request: Request, user: User) -> list[str]:
    if not user.is_demo:
        return [f"user:{user.id}"]

    # The demo account can be recreated. Count both its persistent browser
    # identifier and an IP/browser signature that survives cookie resets.
    visitor = _visitor_id(request)
    if visitor is None:
        raise HTTPException(status_code=401, detail="Please enter the demo again")

    ip = request.client.host if request.client else "unknown"
    agent = request.headers.get("user-agent", "")[:512]
    return [f"demo:{_digest(visitor)}", f"demo-network:{_digest(ip + chr(0) + agent)}"]


def reserve_chat_request(db: Session, request: Request, user: User) -> None:
    """Atomically reserve quota in the database, including concurrent requests."""
    day = datetime.now(timezone.utc).date()
    for identity in _identities(request, user):
        changed = db.execute(
            update(ChatQuota)
            .where(ChatQuota.identity == identity, ChatQuota.day == day, ChatQuota.count < DAILY_LIMIT)
            .values(count=ChatQuota.count + 1)
        ).rowcount
        if changed:
            continue
        try:
            # A savepoint lets a concurrent first insert lose safely, then retry.
            with db.begin_nested():
                db.add(ChatQuota(identity=identity, day=day, count=1))
                db.flush()
        except IntegrityError:
            changed = db.execute(
                update(ChatQuota)
                .where(ChatQuota.identity == identity, ChatQuota.day == day, ChatQuota.count < DAILY_LIMIT)
                .values(count=ChatQuota.count + 1)
            ).rowcount
            if not changed:
                db.rollback()
                raise HTTPException(status_code=429, detail="Assistant daily limit reached (5 requests). Try again tomorrow UTC.")
    db.commit()

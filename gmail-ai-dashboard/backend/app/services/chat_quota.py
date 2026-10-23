"""Reserve one of ten daily assistant requests before running the agent."""

import hashlib
import hmac
from datetime import datetime, timezone

from fastapi import HTTPException, Request
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.chat_quota import ChatQuota
from app.models.email import User

DAILY_LIMIT = 10


def _digest(value: str) -> str:
    return hmac.new(settings.session_secret.encode(), value.encode(), hashlib.sha256).hexdigest()


def _identities(request: Request, user: User) -> list[str]:
    if not user.is_demo:
        return [f"user:{user.id}"]

    # The demo account is shared. A browser token distinguishes visitors on a
    # shared network; the network/browser signature also survives cookie resets.
    raw_cookie = request.cookies.get("gmail_ai_demo_visitor", "")
    visitor, separator, signature = raw_cookie.partition(".")
    if not separator or not visitor or len(visitor) > 128 or not hmac.compare_digest(signature, _digest(visitor)):
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
                raise HTTPException(status_code=429, detail="Assistant daily limit reached (10 requests). Try again tomorrow UTC.")
    db.commit()

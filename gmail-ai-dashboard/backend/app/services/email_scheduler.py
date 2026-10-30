"""Small in-process worker for approved, scheduled Gmail deliveries."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from sqlalchemy import select

from app.database.connection import SessionLocal, CentralSessionLocal
from app.services.developer_settings import decrypt, database_url_context
from app.models.email import User
from app.models.rag import EmailSendApproval
from app.services.gmail_service import send_gmail_message

logger = logging.getLogger(__name__)

_worker_lock = threading.Lock()
_worker_started = False


def deliver_due_scheduled_emails() -> int:
    """Claim and deliver due messages, returning the number sent."""
    delivered = 0
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        statement = (select(EmailSendApproval)
            .where(
                EmailSendApproval.status == "SCHEDULED",
                EmailSendApproval.scheduled_for.is_not(None),
                EmailSendApproval.scheduled_for <= now,
            )
            .order_by(EmailSendApproval.scheduled_for)
            .limit(20))
        if database_url_context.get() is None:
            statement = statement.join(User, User.id == EmailSendApproval.user_id).where(
                User.developer_database_url.is_(None))
        approvals = db.scalars(statement).all()
        for approval in approvals:
            approval.status = "SENDING"
            db.commit()
            user = db.get(User, approval.user_id)
            try:
                if user is None:
                    raise ValueError("The sending user no longer exists")
                result = send_gmail_message(
                    db,
                    user,
                    approval.recipient_email,
                    approval.subject,
                    approval.draft_body,
                    approval.cc,
                    approval.bcc,
                )
                approval.status = "SENT"
                approval.sent_at = datetime.now(timezone.utc)
                approval.delivery_message_id = result.get("id") if isinstance(result, dict) else None
                db.commit()
                delivered += 1
            except Exception as exc:
                approval.status = "SCHEDULED"
                db.commit()
                logger.warning("Scheduled Gmail delivery failed for %s: %s", approval.id, type(exc).__name__)
    if database_url_context.get() is None:
        with CentralSessionLocal() as central:
            personal_urls = [decrypt(value) for value in central.scalars(
                select(User.developer_database_url).where(
                    User.developer_database_url.is_not(None))).all()]
        for url in personal_urls:
            token = database_url_context.set(url)
            try:
                delivered += deliver_due_scheduled_emails()
            except Exception as exc:
                logger.warning("Personal scheduled delivery failed: %s", type(exc).__name__)
            finally:
                database_url_context.reset(token)
    return delivered


def start_global_email_delivery_daemon() -> None:
    """Start one bounded polling worker per API process."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True

    def loop() -> None:
        while True:
            try:
                deliver_due_scheduled_emails()
            except Exception:
                logger.exception("Scheduled Gmail delivery worker failed")
            threading.Event().wait(15)

    threading.Thread(target=loop, name="gmail-scheduled-delivery", daemon=True).start()

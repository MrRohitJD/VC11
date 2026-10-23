"""Read-only Gmail MCP server, launched for one authenticated application user.

The caller cannot choose a user/account through tool arguments. OAuth credentials
stay in the application's database and are never returned through MCP.
"""
import os
import time
from datetime import datetime, timezone
from uuid import UUID

from mcp.server.fastmcp import FastMCP
from sqlalchemy import delete, select
from sqlalchemy.orm import load_only

from app.database.connection import SessionLocal
from app.models.email import Email, EmailLabel, GmailAccount, Label, User
from app.rag.mailbox_scope import NON_PRIMARY_LABELS
from app.core.config import settings
from app.rag.service_errors import error_code
from app.services.gmail_service import build_gmail_service_for_user, _parse_message

mcp = FastMCP("Primary Gmail Search")


def fetch_page(service, items):
    """Use Gmail's batch transport instead of one network round trip per email."""
    messages, errors = {}, []
    def callback(request_id, response, exception):
        if exception is not None:
            errors.append(exception)
        elif response:
            messages[request_id] = response
    batch = service.new_batch_http_request(callback=callback)
    for item in items:
        batch.add(service.users().messages().get(userId='me', id=item['id'], format='full'), request_id=item['id'])
    if items:
        batch.execute()
    if errors:
        raise errors[0]
    if len(messages) != len(items):
        raise ValueError('Incomplete Gmail batch response')
    return [messages[item['id']] for item in items]


def search_primary(db, user_id, query, max_messages):
    deadline = time.monotonic() + max(1, settings.gmail_mcp_timeout_seconds - 20)
    user = db.scalar(select(User).options(load_only(User.id)).where(User.id == UUID(str(user_id))))
    if user is None:
        raise ValueError("Authenticated user is unavailable")
    account = db.scalar(select(GmailAccount).where(GmailAccount.user_id == user.id))
    if account is None:
        raise ValueError("No linked Gmail account")
    service = build_gmail_service_for_user(db, user)
    token = None
    ids = []
    complete = True
    scanned = 0
    failure = ''
    while True:
        if time.monotonic() >= deadline:
            complete, failure = False, 'timeout'
            break
        try:
            page = service.users().messages().list(
                userId="me", q=query, maxResults=min(50, max_messages - scanned),
                pageToken=token, includeSpamTrash=False,
            ).execute()
            messages = fetch_page(service, page.get('messages', []))
        except Exception as exc:
            complete, failure = False, error_code(exc)
            break
        scanned += len(page.get('messages', []))
        for message in messages:
            labels = set(message.get("labelIds", []))
            excluded = {*NON_PRIMARY_LABELS, "SPAM", "TRASH", "DRAFT"}
            if not labels.intersection({"CATEGORY_PERSONAL", "INBOX"}) or labels.intersection(excluded):
                continue
            record = _parse_message(message)
            email = db.scalar(select(Email).where(
                Email.gmail_account_id == account.id, Email.gmail_message_id == message["id"]))
            if email is None:
                email = Email(gmail_account_id=account.id, gmail_message_id=message["id"])
                db.add(email)
            for attr, key in [("subject", "subject"), ("body", "body"), ("body_html", "body_html"),
                              ("snippet", "snippet"), ("from_email", "from"), ("to_email", "to"),
                              ("thread_id", "threadId")]:
                setattr(email, attr, record.get(key))
            email.received_at = datetime.fromtimestamp(int(message["internalDate"]) / 1000, timezone.utc)
            email.is_read = "UNREAD" not in labels
            db.flush()
            db.execute(delete(EmailLabel).where(EmailLabel.email_id == email.id))
            for gmail_label in labels:
                label = db.scalar(select(Label).where(
                    Label.gmail_account_id == account.id, Label.label_id == gmail_label))
                if label is None:
                    label = Label(gmail_account_id=account.id, label_id=gmail_label, name=gmail_label)
                    db.add(label)
                    db.flush()
                db.add(EmailLabel(email_id=email.id, label_id=label.id, gmail_account_id=account.id))
            ids.append(str(email.id))
        # Completed pages survive a later timeout or network failure.
        db.commit()
        token = page.get("nextPageToken")
        if not token:
            break
        if scanned >= max_messages:
            complete = False
            break
    db.commit()
    return {"email_ids": ids, "complete": complete, "source": "gmail_mcp", "error_code": failure}


@mcp.tool()
def search_primary_emails(query: str, max_messages: int = 1000) -> dict:
    """Search and read Primary emails; return persisted application email IDs."""
    if not 1 <= max_messages <= 10000:
        raise ValueError("max_messages must be between 1 and 10000")
    user_id = UUID(os.environ["GMAIL_MCP_USER_ID"])
    with SessionLocal() as db:
        try:
            return search_primary(db, user_id, query, max_messages)
        except Exception as exc:
            db.rollback()
            return {"email_ids": [], "complete": False, "source": "gmail_mcp", "error_code": error_code(exc)}


if __name__ == "__main__":
    mcp.run(transport="stdio")

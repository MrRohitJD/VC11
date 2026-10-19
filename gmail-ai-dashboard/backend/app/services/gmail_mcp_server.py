"""Read-only Gmail MCP server, launched for one authenticated application user.

The caller cannot choose a user/account through tool arguments. OAuth credentials
stay in the application's database and are never returned through MCP.
"""
import os
from datetime import datetime, timezone
from uuid import UUID

from mcp.server.fastmcp import FastMCP
from sqlalchemy import delete, select

from app.database.connection import SessionLocal
from app.models.email import Email, EmailLabel, GmailAccount, Label, User
from app.rag.mailbox_scope import NON_PRIMARY_LABELS
from app.services.gmail_service import build_gmail_service_for_user, _parse_message, _call_with_retry

mcp = FastMCP("Primary Gmail Search")


def search_primary(db, user_id, query, max_messages):
    user = db.get(User, UUID(str(user_id)))
    if user is None:
        raise ValueError("Authenticated user is unavailable")
    account = db.scalar(select(GmailAccount).where(GmailAccount.user_id == user.id))
    if account is None:
        raise ValueError("No linked Gmail account")
    service = build_gmail_service_for_user(db, user)
    token = None
    ids = []
    complete = True
    while True:
        page = _call_with_retry(lambda: service.users().messages().list(
            userId="me", q=query, maxResults=min(100, max_messages - len(ids)),
            pageToken=token, includeSpamTrash=False,
        ).execute())
        for item in page.get("messages", []):
            message = _call_with_retry(lambda: service.users().messages().get(
                userId="me", id=item["id"], format="full",
            ).execute())
            labels = set(message.get("labelIds", []))
            excluded = {*NON_PRIMARY_LABELS, "SPAM", "TRASH", "DRAFT"}
            if not labels.intersection({"CATEGORY_PERSONAL", "INBOX"}) or labels.intersection(excluded):
                continue
            record = _parse_message(message)
            email = db.scalar(select(Email).where(
                Email.gmail_account_id == account.id, Email.gmail_message_id == item["id"]))
            if email is None:
                email = Email(gmail_account_id=account.id, gmail_message_id=item["id"])
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
        token = page.get("nextPageToken")
        if not token:
            break
        if len(ids) >= max_messages:
            complete = False
            break
    db.commit()
    return {"email_ids": ids, "complete": complete, "source": "gmail_mcp"}


@mcp.tool()
def search_primary_emails(query: str, max_messages: int = 1000) -> dict:
    """Search and read Primary emails; return persisted application email IDs."""
    if not 1 <= max_messages <= 10000:
        raise ValueError("max_messages must be between 1 and 10000")
    user_id = UUID(os.environ["GMAIL_MCP_USER_ID"])
    with SessionLocal() as db:
        return search_primary(db, user_id, query, max_messages)


if __name__ == "__main__":
    mcp.run(transport="stdio")

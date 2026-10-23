"""Gmail-to-LangChain document adapter.

Converts Gmail messages from database / Gmail API into LangChain Document objects,
preserving all required envelope metadata, identifiers, and audit fields.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, List, Optional
from uuid import UUID

from langchain_core.documents import Document

from app.models.email import Email, GmailAccount
from app.services.email_chunker import format_embedding_text


def compute_content_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of email text content."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class GmailDocumentAdapter:
    """Converts email entities into standard LangChain Document objects."""

    @staticmethod
    def email_to_document(
        email: Email,
        user_id: Optional[UUID] = None,
        account: Optional[GmailAccount] = None,
    ) -> Document:
        """Convert an SQLAlchemy Email instance into a LangChain Document."""
        resolved_user_id = ""
        if user_id:
            resolved_user_id = str(user_id)
        elif account and account.user_id:
            resolved_user_id = str(account.user_id)
        elif email.gmail_account and email.gmail_account.user_id:
            resolved_user_id = str(email.gmail_account.user_id)

        # Extract label names
        label_names: List[str] = []
        if hasattr(email, "labels") and email.labels:
            for el in email.labels:
                if hasattr(el, "label") and el.label and hasattr(el.label, "name"):
                    label_names.append(el.label.name)
                elif hasattr(el, "name"):
                    label_names.append(el.name)

        # Attachment metadata
        attachments_meta = []
        if email.attachments_meta:
            if isinstance(email.attachments_meta, list):
                attachments_meta = email.attachments_meta
            elif isinstance(email.attachments_meta, dict):
                attachments_meta = [email.attachments_meta]

        # Prepare text representation
        body_content = email.body or ""
        subject_content = email.subject or ""
        full_text = format_embedding_text(
            subject=subject_content,
            chunk_text=body_content,
            sender_name=email.sender_name,
            sender_email=email.from_email,
            gmail_message_id=email.gmail_message_id,
            received_at=email.received_at,
            recipient_emails=email.to_email,
        )

        content_hash = compute_content_hash(full_text)

        metadata: dict[str, Any] = {
            "user_id": resolved_user_id,
            "gmail_account_id": str(email.gmail_account_id) if email.gmail_account_id else "",
            "email_id": str(email.id),
            "thread_id": email.thread_id or "",
            "gmail_message_id": email.gmail_message_id or "",
            "subject": subject_content,
            "sender": email.from_email or "",
            "sender_email": email.from_email or "",
            "from_email": email.from_email or "",
            "sender_name": email.sender_name or "",
            "recipients": email.to_email or "",
            "cc": email.cc or "",
            "bcc": email.bcc or "",
            "date": email.received_at.isoformat() if email.received_at else "",
            "received_at": email.received_at.isoformat() if email.received_at else "",
            "labels": json.dumps(label_names),
            "mime_type": "text/plain",
            "has_attachment": bool(email.has_attachment),
            "attachment_metadata": json.dumps(attachments_meta),
            "source": f"gmail:{email.gmail_message_id}",
            "content_hash": content_hash,
            "is_read": bool(email.is_read),
            "is_starred": bool(email.is_starred),
        }

        return Document(page_content=full_text, metadata=metadata)

    @staticmethod
    def emails_to_documents(
        emails: List[Email],
        user_id: Optional[UUID] = None,
        account: Optional[GmailAccount] = None,
    ) -> List[Document]:
        """Convert a list of emails to LangChain documents."""
        return [
            GmailDocumentAdapter.email_to_document(email, user_id=user_id, account=account)
            for email in emails
        ]

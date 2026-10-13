"""LangChain tool definitions for the Gmail RAG Assistant.

Implements official LangChain `@tool` decorator with validated arguments,
typed schemas, error handling, and authorization scoping.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.tools import tool
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models.email import AiCategory, Email, EmailClassification, EmailLabel, GmailAccount, Label
from app.models.rag import Attachment, EmailSendApproval
from app.rag.retrieval import (
    ChromaSemanticRetriever,
    GmailCustomHybridRetriever,
    MultiQueryRetriever,
    default_retriever,
    documents_to_citations,
)

logger = logging.getLogger(__name__)


def create_gmail_tools(db: Session, user_id: UUID | str) -> List[Any]:
    """Factory creating user-scoped LangChain tools for agent execution."""
    user_uuid = UUID(str(user_id))

    @tool
    def search_emails(query: str, top_k: int = 5) -> str:
        """Search emails semantically using ChromaDB vector retrieval."""
        retriever = ChromaSemanticRetriever(
            user_id=str(user_uuid),
            top_k=top_k,
        )
        docs = retriever.invoke(query)
        citations = documents_to_citations(docs)
        if not citations:
            return "No matching emails found in semantic vector search."

        results = []
        for c in citations:
            results.append({
                "email_id": c.email_id,
                "subject": c.subject,
                "sender": c.sender,
                "date": c.date,
                "snippet": c.snippet,
                "relevance_score": c.score,
            })
        return json.dumps(results, indent=2)

    @tool
    def get_email_by_id(email_id: str) -> str:
        """Fetch the full content, sender, subject, and metadata of a specific email by ID."""
        try:
            eid = UUID(str(email_id).strip())
        except Exception:
            return f"Invalid email UUID format: {email_id}"

        email = db.scalar(
            select(Email)
            .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(Email.id == eid, GmailAccount.user_id == user_uuid)
        )
        if not email:
            return f"Email {email_id} not found or unauthorized."

        return json.dumps({
            "id": str(email.id),
            "subject": email.subject or "",
            "from": email.from_email or "",
            "to": email.to_email or "",
            "date": email.received_at.isoformat() if email.received_at else "",
            "snippet": email.snippet or "",
            "body": (email.body or "")[:2000],
            "is_read": email.is_read,
        }, indent=2)

    @tool
    def search_email_threads(thread_id: str) -> str:
        """Retrieve all emails belonging to an email thread."""
        emails = db.scalars(
            select(Email)
            .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(Email.thread_id == thread_id, GmailAccount.user_id == user_uuid)
            .order_by(Email.received_at.asc().nullslast())
        ).all()

        if not emails:
            return f"No thread found with ID {thread_id}."

        thread_messages = []
        for e in emails:
            thread_messages.append({
                "email_id": str(e.id),
                "from": e.from_email or "",
                "date": e.received_at.isoformat() if e.received_at else "",
                "subject": e.subject or "",
                "snippet": e.snippet or "",
            })
        return json.dumps(thread_messages, indent=2)

    @tool
    def retrieve_email_context(query: str, top_k: int = 5) -> str:
        """Perform hybrid retrieval (dense ChromaDB vector + PostgreSQL text) for RAG context."""
        retriever = GmailCustomHybridRetriever(
            user_id=str(user_uuid),
            db_session=db,
            top_k=top_k,
        )
        docs = retriever.invoke(query)
        citations = documents_to_citations(docs)
        if not citations:
            return "No relevant context found."

        formatted = []
        for c in citations:
            formatted.append(f"[{c.subject}] from {c.sender} ({c.date}): {c.snippet}")
        return "\n\n".join(formatted)

    @tool
    def search_emails_multi_query(query: str, top_k: int = 5) -> str:
        """Generate alternative search perspectives, execute multi-query retrieval, and fuse results via RRF."""
        base_retr = GmailCustomHybridRetriever(
            user_id=str(user_uuid),
            db_session=db,
            top_k=top_k,
        )
        mq_retriever = MultiQueryRetriever(
            base_retriever=base_retr,
            query_count=3,
        )
        docs = mq_retriever.invoke(query)
        citations = documents_to_citations(docs)
        if not citations:
            return "No matching emails found across multi-query perspectives."

        return json.dumps([
            {
                "email_id": c.email_id,
                "subject": c.subject,
                "sender": c.sender,
                "date": c.date,
                "snippet": c.snippet,
                "score": c.score,
            }
            for c in citations
        ], indent=2)

    @tool
    def filter_emails_hybrid(
        query: str,
        sender: Optional[str] = None,
        label: Optional[str] = None,
        is_read: Optional[bool] = None,
        thread_id: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """Execute custom hybrid retrieval combining ChromaDB semantic search with relational filters."""
        retriever = GmailCustomHybridRetriever(
            user_id=str(user_uuid),
            db_session=db,
            top_k=top_k,
            sender=sender,
            label=label,
            is_read=is_read,
            thread_id=thread_id,
        )
        docs = retriever.invoke(query)
        citations = documents_to_citations(docs)
        if not citations:
            return "No matching emails found matching the hybrid criteria."
        return json.dumps([c.__dict__ for c in citations], indent=2)

    @tool
    def filter_emails_by_date(start_date: str, end_date: str, limit: int = 5) -> str:
        """Filter emails received within a date range (YYYY-MM-DD format)."""
        try:
            s_dt = datetime.fromisoformat(start_date)
            e_dt = datetime.fromisoformat(end_date)
        except Exception:
            return "Invalid date format. Use YYYY-MM-DD."

        citations = default_retriever.retrieve_structured_sql(
            db=db,
            user_id=user_uuid,
            start_date=s_dt,
            end_date=e_dt,
            limit=limit,
        )
        return json.dumps([c.__dict__ for c in citations], indent=2)

    @tool
    def filter_emails_by_sender(sender: str, limit: int = 5) -> str:
        """Filter emails sent by a specific email address or domain name."""
        citations = default_retriever.retrieve_structured_sql(
            db=db,
            user_id=user_uuid,
            sender=sender,
            limit=limit,
        )
        return json.dumps([c.__dict__ for c in citations], indent=2)

    @tool
    def filter_emails_by_label(label: str, limit: int = 5) -> str:
        """Filter emails categorized under an AI classification or Gmail label (e.g. WORK, JOB, BILLS)."""
        citations = default_retriever.retrieve_structured_sql(
            db=db,
            user_id=user_uuid,
            label=label,
            limit=limit,
        )
        return json.dumps([c.__dict__ for c in citations], indent=2)

    @tool
    def get_email_summary(email_id: str) -> str:
        """Generate a concise summary of an email by ID."""
        try:
            eid = UUID(str(email_id).strip())
        except Exception:
            return f"Invalid email UUID format: {email_id}"

        email = db.scalar(
            select(Email)
            .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(Email.id == eid, GmailAccount.user_id == user_uuid)
        )
        if not email:
            return f"Email {email_id} not found."

        snippet = email.snippet or email.body or "Empty body"
        return f"Summary of '{email.subject}': {snippet[:250]}..."

    @tool
    def classify_email(email_id: str) -> str:
        """Get the current AI category label for a given email."""
        try:
            eid = UUID(str(email_id).strip())
        except Exception:
            return f"Invalid email UUID: {email_id}"

        classification = db.scalar(
            select(AiCategory.name)
            .join(EmailClassification, EmailClassification.category_id == AiCategory.id)
            .where(EmailClassification.email_id == eid, EmailClassification.is_current.is_(True))
        )
        return classification or "Uncategorized"

    @tool
    def get_email_attachments(email_id: str) -> str:
        """Retrieve metadata and extracted text for attachments belonging to an email."""
        try:
            eid = UUID(str(email_id).strip())
        except Exception:
            return f"Invalid email UUID: {email_id}"

        attachments = db.scalars(
            select(Attachment).where(Attachment.email_id == eid)
        ).all()
        if not attachments:
            return "No attachments registered for this email."

        res = []
        for a in attachments:
            res.append({
                "filename": a.filename,
                "mime_type": a.mime_type,
                "size_bytes": a.file_size_bytes,
                "status": a.extraction_status,
                "extracted_snippet": (a.extracted_text or "")[:200],
            })
        return json.dumps(res, indent=2)

    @tool
    def draft_email(recipient: str, subject: str, body: str) -> str:
        """Create a drafted outbound email requiring user confirmation before sending."""
        approval = EmailSendApproval(
            user_id=user_uuid,
            recipient_email=recipient,
            subject=subject,
            draft_body=body,
            status="PENDING",
        )
        db.add(approval)
        db.commit()
        return (
            f"Draft created successfully (Approval ID: {approval.id}). "
            f"Subject: '{subject}' to '{recipient}'. "
            f"NOTE: Requires human confirmation before sending."
        )

    @tool
    def send_email(approval_id: str) -> str:
        """Send a drafted email ONLY if explicit user approval was registered.

        Enforces human-in-the-loop security.
        """
        try:
            aid = UUID(str(approval_id).strip())
        except Exception:
            return f"Invalid approval ID: {approval_id}"

        approval = db.scalar(
            select(EmailSendApproval).where(
                EmailSendApproval.id == aid,
                EmailSendApproval.user_id == user_uuid,
            )
        )
        if not approval:
            return f"Approval request {approval_id} not found."

        if approval.status != "APPROVED":
            return (
                f"Cannot send email. Status is currently '{approval.status}'. "
                f"Human approval must be granted explicitly by the user."
            )

        # Mark sent
        approval.status = "SENT"
        approval.sent_at = datetime.utcnow()
        db.commit()
        return f"Email successfully sent to {approval.recipient_email}."

    return [
        search_emails,
        search_emails_multi_query,
        get_email_by_id,
        search_email_threads,
        retrieve_email_context,
        filter_emails_hybrid,
        filter_emails_by_date,
        filter_emails_by_sender,
        filter_emails_by_label,
        get_email_summary,
        classify_email,
        get_email_attachments,
        draft_email,
        send_email,
    ]

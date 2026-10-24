"""LangChain tool definitions for the Gmail RAG Assistant.

Implements official LangChain `@tool` decorator with validated arguments,
typed schemas, error handling, and authorization scoping.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.tools import tool
from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.email import AiCategory, Email, EmailClassification, EmailLabel, GmailAccount, Label, User
from app.models.rag import Attachment, EmailSendApproval
from app.rag.mailbox_scope import mailbox_email_condition
from app.services.gmail_service import normalize_recipient_header, send_gmail_message
from app.rag.retrieval import (
    PostgresSemanticRetriever,
    ConditionalMultiQueryRetriever,
    GmailCustomHybridRetriever,
    MultiQueryRetriever,
    default_retriever,
    documents_to_citations,
)

logger = logging.getLogger(__name__)


def sanitize_tool_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize arguments for safe audit logging, stripping bodies, OTPs, and tokens."""
    sensitive_keys = {
        "body", "body_text", "body_html", "draft_body",
        "token", "access_token", "refresh_token", "otp", "password", "secret",
    }
    sanitized = {}
    for k, v in args.items():
        if k in sensitive_keys:
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, str) and len(v) > 100:
            sanitized[k] = v[:100] + "..."
        else:
            sanitized[k] = v
    return sanitized


def parse_relative_or_explicit_date_range(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    range_description: Optional[str] = None,
    now: Optional[datetime] = None,
) -> tuple[Optional[datetime], Optional[datetime]]:
    """Parse explicit or relative date ranges with boundary and timezone handling.

    Supports:
    - Relative ranges: 'last month', 'last 2 months', 'last 3 months', 'last N days/weeks/years', 'today', 'yesterday'
    - Explicit ranges: 'YYYY-MM-DD' or ISO-8601 strings
    """
    now = now or datetime.now(timezone.utc)
    target_str = (range_description or start_date or "").strip().lower()

    # Relative patterns
    m_months = re.search(r"(?:last|past)\s+(\d+)\s+months?", target_str)
    if m_months:
        count = int(m_months.group(1))
        return now - timedelta(days=count * 30), now

    if "last month" in target_str or "past month" in target_str:
        return now - timedelta(days=30), now

    m_days = re.search(r"(?:last|past)\s+(\d+)\s+days?", target_str)
    if m_days:
        count = int(m_days.group(1))
        return now - timedelta(days=count), now

    m_weeks = re.search(r"(?:last|past)\s+(\d+)\s+weeks?", target_str)
    if m_weeks:
        count = int(m_weeks.group(1))
        return now - timedelta(days=count * 7), now

    if "last week" in target_str or "past week" in target_str:
        return now - timedelta(days=7), now

    if "last year" in target_str or "past year" in target_str:
        return now - timedelta(days=365), now

    if "today" in target_str:
        s_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return s_dt, now

    if "yesterday" in target_str:
        yest = now - timedelta(days=1)
        s_dt = yest.replace(hour=0, minute=0, second=0, microsecond=0)
        e_dt = yest.replace(hour=23, minute=59, second=59, microsecond=999999)
        return s_dt, e_dt

    # Explicit date strings
    s_dt = None
    e_dt = None

    if start_date:
        try:
            clean_s = start_date.strip()
            if "t" in clean_s.lower():
                s_dt = datetime.fromisoformat(clean_s)
            else:
                d = datetime.strptime(clean_s[:10], "%Y-%m-%d")
                s_dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
            if s_dt.tzinfo is None:
                s_dt = s_dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass

    if end_date:
        try:
            clean_e = end_date.strip()
            if "t" in clean_e.lower():
                e_dt = datetime.fromisoformat(clean_e)
            else:
                d = datetime.strptime(clean_e[:10], "%Y-%m-%d")
                e_dt = datetime(d.year, d.month, d.day, 23, 59, 59, 999999, tzinfo=timezone.utc)
            if e_dt.tzinfo is None:
                e_dt = e_dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass

    if s_dt and not e_dt:
        e_dt = now

    return s_dt, e_dt


def create_gmail_tools(db: Session, user_id: UUID | str) -> List[Any]:
    """Factory creating user-scoped tools across all non-junk Gmail categories."""
    user_uuid = UUID(str(user_id))

    @tool
    def search_emails(query: str, top_k: int = 5) -> str:
        """Search emails semantically using PostgreSQL pgvector retrieval."""
        sanitized = sanitize_tool_args({"query": query, "top_k": top_k})
        logger.info("Tool chosen: search_emails with args: %s", sanitized)

        retriever = PostgresSemanticRetriever(
            user_id=str(user_uuid),
            db_session=db,
            primary_only=False,
            top_k=top_k,
        )
        docs = retriever.invoke(query)
        citations = documents_to_citations(docs)
        returned_ids = [c.email_id for c in citations]
        logger.info("Tool search_emails returned %d row(s), email_ids: %s", len(citations), returned_ids)
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
        sanitized = sanitize_tool_args({"email_id": email_id})
        logger.info("Tool chosen: get_email_by_id with args: %s", sanitized)

        try:
            eid = UUID(str(email_id).strip())
        except Exception:
            logger.warning("get_email_by_id invalid UUID: %s", email_id)
            return f"Invalid email UUID format: {email_id}"

        email = db.scalar(
            select(Email)
            .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(Email.id == eid, GmailAccount.user_id == user_uuid)
            .where(mailbox_email_condition())
        )
        if not email:
            logger.info("Tool get_email_by_id returned 0 row(s) for ID: %s", email_id)
            return f"Email {email_id} not found or unauthorized."

        logger.info("Tool get_email_by_id returned 1 row, email_id: %s", str(email.id))
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
        sanitized = sanitize_tool_args({"thread_id": thread_id})
        logger.info("Tool chosen: search_email_threads with args: %s", sanitized)

        emails = db.scalars(
            select(Email)
            .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(Email.thread_id == thread_id, GmailAccount.user_id == user_uuid)
            .where(mailbox_email_condition())
            .order_by(Email.received_at.asc().nullslast())
        ).all()

        returned_ids = [str(e.id) for e in emails]
        logger.info("Tool search_email_threads returned %d row(s), email_ids: %s", len(emails), returned_ids)
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
        """Perform hybrid retrieval (dense pgvector + PostgreSQL text) for RAG context."""
        sanitized = sanitize_tool_args({"query": query, "top_k": top_k})
        logger.info("Tool chosen: retrieve_email_context with args: %s", sanitized)

        base_retriever = GmailCustomHybridRetriever(
            user_id=str(user_uuid),
            db_session=db,
            primary_only=False,
            top_k=top_k,
        )
        retriever = ConditionalMultiQueryRetriever(
            base_retriever=base_retriever,
            query_count=3,
        )
        docs = retriever.invoke(query)
        citations = documents_to_citations(docs)
        returned_ids = [c.email_id for c in citations]
        logger.info("Tool retrieve_email_context returned %d row(s), email_ids: %s", len(citations), returned_ids)
        if not citations:
            return "No relevant context found."

        # Return JSON list of citations for easy consumption by agent
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
    def search_emails_multi_query(query: str, top_k: int = 5) -> str:
        """Generate alternative search perspectives, execute multi-query retrieval, and fuse results via RRF."""
        sanitized = sanitize_tool_args({"query": query, "top_k": top_k})
        logger.info("Tool chosen: search_emails_multi_query with args: %s", sanitized)

        base_retr = GmailCustomHybridRetriever(
            user_id=str(user_uuid),
            db_session=db,
            primary_only=False,
            top_k=top_k,
        )
        mq_retriever = MultiQueryRetriever(
            base_retriever=base_retr,
            query_count=3,
        )
        docs = mq_retriever.invoke(query)
        citations = documents_to_citations(docs)
        returned_ids = [c.email_id for c in citations]
        logger.info("Tool search_emails_multi_query returned %d row(s), email_ids: %s", len(citations), returned_ids)
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
        query: str = "",
        sender: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        label: Optional[str] = None,
        is_read: Optional[bool] = None,
        thread_id: Optional[str] = None,
        top_k: int = 10,
    ) -> str:
        """Execute custom hybrid retrieval combining pgvector semantic search with PostgreSQL relational filters."""
        s_dt, e_dt = parse_relative_or_explicit_date_range(start_date=start_date, end_date=end_date)
        sanitized = sanitize_tool_args({
            "query": query,
            "sender": sender,
            "start_date": start_date,
            "end_date": end_date,
            "label": label,
            "is_read": is_read,
            "thread_id": thread_id,
            "top_k": top_k,
        })
        logger.info("Tool chosen: filter_emails_hybrid with args: %s", sanitized)

        retriever = GmailCustomHybridRetriever(
            user_id=str(user_uuid),
            db_session=db,
            primary_only=False,
            top_k=top_k,
            sender=sender,
            start_date=s_dt,
            end_date=e_dt,
            label=label,
            is_read=is_read,
            thread_id=thread_id,
        )
        docs = retriever.invoke(query)
        citations = documents_to_citations(docs)
        returned_ids = [c.email_id for c in citations]
        logger.info("Tool filter_emails_hybrid returned %d row(s), email_ids: %s", len(citations), returned_ids)
        if not citations:
            return "No matching emails found matching the hybrid criteria."
        return json.dumps([c.__dict__ for c in citations], indent=2)

    @tool
    def filter_emails_by_date(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        range_description: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Filter emails received within a date range (YYYY-MM-DD format or relative like 'last 2 months', 'last 3 months')."""
        s_dt, e_dt = parse_relative_or_explicit_date_range(
            start_date=start_date,
            end_date=end_date,
            range_description=range_description,
        )
        sanitized = sanitize_tool_args({
            "start_date": start_date,
            "end_date": end_date,
            "range_description": range_description,
            "limit": limit,
        })
        logger.info("Tool chosen: filter_emails_by_date with args: %s", sanitized)

        citations = default_retriever.retrieve_structured_sql(
            db=db,
            user_id=user_uuid,
            start_date=s_dt,
            end_date=e_dt,
            limit=limit,
            primary_only=False,
        )
        returned_ids = [c.email_id for c in citations]
        logger.info("Tool filter_emails_by_date returned %d row(s), email_ids: %s", len(citations), returned_ids)
        return json.dumps([c.__dict__ for c in citations], indent=2)

    @tool
    def filter_emails_by_sender(sender: str, limit: int = 10) -> str:
        """Filter emails sent by a specific email address, person name, or domain name."""
        clean_sender = (sender or "").strip()
        sanitized = sanitize_tool_args({"sender": clean_sender, "limit": limit})
        logger.info("Tool chosen: filter_emails_by_sender with args: %s", sanitized)

        citations = default_retriever.retrieve_structured_sql(
            db=db,
            user_id=user_uuid,
            sender=clean_sender,
            limit=limit,
            primary_only=False,
        )
        returned_ids = [c.email_id for c in citations]
        logger.info("Tool filter_emails_by_sender returned %d row(s), email_ids: %s", len(citations), returned_ids)
        return json.dumps([c.__dict__ for c in citations], indent=2)

    @tool
    def filter_emails_by_label(label: str, limit: int = 10) -> str:
        """Filter emails categorized under an AI classification or Gmail label (e.g. WORK, JOB, BILLS)."""
        clean_lbl = (label or "").strip()
        sanitized = sanitize_tool_args({"label": clean_lbl, "limit": limit})
        logger.info("Tool chosen: filter_emails_by_label with args: %s", sanitized)

        citations = default_retriever.retrieve_structured_sql(
            db=db,
            user_id=user_uuid,
            label=clean_lbl,
            limit=limit,
            primary_only=False,
        )
        returned_ids = [c.email_id for c in citations]
        logger.info("Tool filter_emails_by_label returned %d row(s), email_ids: %s", len(citations), returned_ids)
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
            .where(mailbox_email_condition())
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
            .join(Email, Email.id == EmailClassification.email_id)
            .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(EmailClassification.email_id == eid, EmailClassification.is_current.is_(True))
            .where(GmailAccount.user_id == user_uuid, mailbox_email_condition())
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
            select(Attachment)
            .join(Email, Email.id == Attachment.email_id)
            .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(Attachment.email_id == eid, GmailAccount.user_id == user_uuid, mailbox_email_condition())
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
    def draft_email(recipient: str, subject: str, body: str, cc: str = "", bcc: str = "") -> str:
        """Create a drafted outbound email requiring user confirmation before sending."""
        try:
            recipient = normalize_recipient_header(recipient, "recipient", required=True)
            cc = normalize_recipient_header(cc, "Cc")
            bcc = normalize_recipient_header(bcc, "Bcc")
        except ValueError as exc:
            return f"Draft could not be created: {exc}"
        sanitized = sanitize_tool_args({"recipient": recipient, "cc": cc, "bcc": bcc, "subject": subject, "body": body})
        logger.info("Tool chosen: draft_email with args: %s", sanitized)
        approval = EmailSendApproval(
            user_id=user_uuid,
            recipient_email=recipient,
            cc=cc or None,
            bcc=bcc or None,
            subject=subject,
            draft_body=body,
            status="PENDING",
        )
        db.add(approval)
        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            logger.exception("Could not persist drafted email approval")
            return (
                "I couldn't save the draft because the email database schema is out of date. "
                "Please run `alembic upgrade head` from the backend directory and try again."
            )
        return (
            f"Draft created successfully (Approval ID: {approval.id}). "
            f"Subject: '{subject}' to '{recipient}'. "
            f"NOTE: Requires human confirmation before sending."
        )

    @tool
    def schedule_email(
        recipient: str,
        subject: str,
        body: str,
        scheduled_for: str,
        cc: str = "",
        bcc: str = "",
    ) -> str:
        """Create a future email delivery requiring explicit user approval."""
        try:
            recipient = normalize_recipient_header(recipient, "recipient", required=True)
            cc = normalize_recipient_header(cc, "Cc")
            bcc = normalize_recipient_header(bcc, "Bcc")
        except ValueError as exc:
            return f"Scheduled draft could not be created: {exc}"
        sanitized = sanitize_tool_args({
            "recipient": recipient, "cc": cc, "bcc": bcc, "subject": subject,
            "body": body, "scheduled_for": scheduled_for,
        })
        logger.info("Tool chosen: schedule_email with args: %s", sanitized)
        try:
            value = str(scheduled_for or "").strip().replace("Z", "+00:00")
            when = datetime.fromisoformat(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            when = when.astimezone(timezone.utc)
        except Exception:
            return "Invalid scheduled_for value. Use an ISO-8601 datetime with timezone."
        if when <= datetime.now(timezone.utc):
            return "The scheduled time must be in the future."
        approval = EmailSendApproval(
            user_id=user_uuid,
            recipient_email=recipient,
            cc=cc or None,
            bcc=bcc or None,
            subject=subject,
            draft_body=body,
            status="PENDING",
            scheduled_for=when,
        )
        db.add(approval)
        db.commit()
        return (
            f"Scheduled draft created (Approval ID: {approval.id}) for {when.isoformat()}. "
            "Explicit approval is required before the message is queued for delivery."
        )

    @tool
    def send_email(approval_id: str) -> str:
        """Send a drafted email ONLY if explicit user approval was registered.

        Enforces human-in-the-loop security.
        """
        sanitized = sanitize_tool_args({"approval_id": approval_id})
        logger.info("Tool chosen: send_email with args: %s", sanitized)
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

        if approval.scheduled_for:
            return "This approval is scheduled; approve it through the mailbox UI to queue delivery."

        user = db.get(User, user_uuid)
        if user is None:
            return "Email could not be sent because the authenticated user was not found."
        try:
            delivery = send_gmail_message(
                db,
                user,
                approval.recipient_email,
                approval.subject,
                approval.draft_body,
                approval.cc,
                approval.bcc,
            )
        except Exception as exc:
            logger.warning("Approved Gmail delivery failed: %s", type(exc).__name__)
            return f"Email could not be sent: {exc}"

        approval.status = "SENT"
        approval.sent_at = datetime.now(timezone.utc)
        approval.delivery_message_id = delivery.get("id") if isinstance(delivery, dict) else None
        db.commit()
        return f"Email successfully sent to {approval.recipient_email}."

    tools = [
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
        schedule_email,
        send_email,
    ]
    owner = db.get(User, user_uuid)
    if owner is not None and owner.is_demo:
        # The public demo can use the assistant for mailbox search and
        # summaries, but it must never create or deliver outbound mail.
        outbound_tools = {"draft_email", "schedule_email", "send_email"}
        tools = [candidate for candidate in tools if candidate.name not in outbound_tools]
    return tools

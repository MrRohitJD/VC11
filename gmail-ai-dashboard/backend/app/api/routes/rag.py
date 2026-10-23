"""FastAPI routes for LangChain-native Gmail RAG and AI Assistant.

Provides:
- POST /api/rag/chat: Run 10-pattern Gmail AI Assistant with citations and tool reasoning
- POST /api/rag/index: Trigger incremental ChromaDB indexing for user emails
- GET  /api/rag/status: Diagnostic overview of ChromaDB and PostgreSQL sync status
- GET  /api/rag/sessions: User conversational sessions list
- GET  /api/rag/sessions/{id}: Message history and citation logs for a session
- DELETE /api/rag/sessions/{id}: Delete a chat session
- GET  /api/rag/approvals: List human-in-the-loop pending email draft approvals
- POST /api/rag/approvals/{id}/approve: Grant human authorization to send email
- POST /api/rag/approvals/{id}/reject: Reject drafted email
- GET  /api/rag/sources/{id}: Detailed email view for cited sources
"""

from __future__ import annotations

import logging
import json
from queue import Queue, Empty
from threading import Thread, Event
from datetime import datetime, timezone
from email.utils import parseaddr
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.connection import get_db, SessionLocal
from app.services.chat_quota import reserve_chat_request
from app.rag.activity import activity_context
from app.models.email import Email, GmailAccount, User
from app.models.rag import (
    Attachment,
    ChatMessage,
    ChatRetrieval,
    ChatSession,
    EmailChunk,
    EmailEmbedding,
    EmailSendApproval,
)
from app.rag.graph_agent import default_assistant
from app.rag.preprocessing import clean_email_text
from app.rag.query_constraints import mailbox_time_context
from app.rag.indexing import default_pipeline
from app.rag.memory import list_user_sessions
from app.rag.vectorstore import default_vectorstore
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


# ---------------------------------------------------------------------------
# Authentication Helper with Development Fallback
# ---------------------------------------------------------------------------
def get_user_for_rag(request: Request, db: Session = Depends(get_db)) -> User:
    """Never substitute another account when a session is absent or invalid."""
    return get_current_user(request, db)


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., description="User question or instruction")
    session_id: Optional[str] = Field(None, description="Optional existing session UUID")
    top_k: int = Field(5, description="Number of source citations to retrieve")
    filter_dict: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")
    sender: Optional[str] = Field(None, description="Filter by sender email or domain")
    label: Optional[str] = Field(None, description="Filter by label or category")
    is_read: Optional[bool] = Field(None, description="Filter by read status")
    thread_id: Optional[str] = Field(None, description="Filter by thread ID")
    use_multi_query: Optional[bool] = Field(None, description="None selects automatically; true/false overrides query expansion")


class CitationItem(BaseModel):
    email_id: str
    subject: str
    sender: str
    date: str
    snippet: str
    score: float
    key_points: List[str] = Field(default_factory=list, description="Key points extracted from email as single-sentence bullets")


class ChatResponse(BaseModel):
    answer: str
    intent: str
    citations: List[CitationItem]
    steps: List[Dict[str, Any]]
    plan: Optional[List[str]] = None
    approval_required: bool = False
    approval_id: Optional[str] = None
    session_id: str
    message_id: str
    key_points: List[str] = Field(default_factory=list, description="Extracted key points based on user query in single-sentence bullets")
    search_plan: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    trace_url: Optional[str] = None
    tracing_enabled: bool = False


class IndexRequest(BaseModel):
    batch_size: Optional[int] = Field(None, description="Maximum emails to process; omit to process all emails")
    full_reindex: bool = Field(False, description="Whether to reprocess all emails")


class ApprovalActionResponse(BaseModel):
    approval_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
def chat_with_assistant(
    body: ChatRequest,
    request: Request,
    user: User = Depends(get_user_for_rag),
    db: Session = Depends(get_db),
    x_timezone: Optional[str] = Header(None, alias="X-Timezone"),
):
    """Execute grounded question answering or agent execution for the user's emails."""
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    reserve_chat_request(db, request, user)
    return _run_chat(body, user, db, x_timezone)


def _run_chat(body: ChatRequest, user: User, db: Session, x_timezone: Optional[str]) -> ChatResponse:
    try:
        f_dict = dict(body.filter_dict or {})
        if body.sender:
            f_dict["sender"] = body.sender
        if body.label:
            f_dict["label"] = body.label
        if body.is_read is not None:
            f_dict["is_read"] = body.is_read
        if body.thread_id:
            f_dict["thread_id"] = body.thread_id

        with mailbox_time_context(x_timezone):
            agent_res = default_assistant.run(
                query=body.query.strip(),
                db=db,
                user_id=user.id,
                session_id=body.session_id,
                top_k=body.top_k,
                filter_dict=f_dict if f_dict else None,
                use_multi_query=body.use_multi_query,
            )

        citation_items = [
            CitationItem(
                email_id=str(c.email_id),
                subject=c.subject,
                sender=c.sender,
                date=c.date,
                snippet=c.snippet,
                score=round(c.score, 4),
                key_points=getattr(c, "key_points", []) or [],
            )
            for c in agent_res.citations
        ]

        # Extract unique single-sentence key points across all retrieved citations
        extracted_key_points: List[str] = []
        seen_kp = set()
        for c in agent_res.citations:
            for kp in (getattr(c, "key_points", []) or []):
                norm_kp = kp.strip().lower()
                if norm_kp and norm_kp not in seen_kp:
                    seen_kp.add(norm_kp)
                    extracted_key_points.append(kp)

        # Fallback if citations had snippets but no explicit key points
        if not extracted_key_points and agent_res.citations:
            from app.rag.retrieval import split_into_single_sentence_bullets
            for c in agent_res.citations:
                bullets = split_into_single_sentence_bullets(c.snippet, max_points=3)
                for b in bullets:
                    norm_b = b.strip().lower()
                    if norm_b and norm_b not in seen_kp:
                        seen_kp.add(norm_b)
                        extracted_key_points.append(b)

        return ChatResponse(
            answer=agent_res.answer,
            intent=agent_res.intent,
            citations=citation_items,
            steps=agent_res.steps,
            plan=agent_res.plan,
            approval_required=agent_res.approval_required,
            approval_id=agent_res.approval_id,
            session_id=agent_res.session_id or "",
            message_id=agent_res.message_id or "",
            key_points=extracted_key_points,
            search_plan=getattr(agent_res, "search_plan", None),
            trace_id=getattr(agent_res, "trace_id", None),
            trace_url=getattr(agent_res, "trace_url", None),
            tracing_enabled=getattr(agent_res, "tracing_enabled", False),
        )
    except Exception as exc:
        logger.exception("RAG assistant execution error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(exc)}",
        )


@router.post("/chat/stream")
def stream_chat(
    body: ChatRequest,
    request: Request,
    user: User = Depends(get_user_for_rag),
    db: Session = Depends(get_db),
    x_timezone: Optional[str] = Header(None, alias="X-Timezone"),
):
    """Stream observable progress and one final response; never stream reasoning."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    reserve_chat_request(db, request, user)
    user_id = user.id
    events = Queue()
    disconnected = Event()

    def publish(step):
        if not disconnected.is_set():
            events.put(("activity", step))

    def work():
        try:
            # A SQLAlchemy session belongs to one worker, not the request thread.
            with SessionLocal() as db:
                scoped_user = db.get(User, user_id)
                if scoped_user is None:
                    raise ValueError("User session is unavailable")
                with activity_context(publish):
                    response = _run_chat(body, scoped_user, db, x_timezone)
                if not disconnected.is_set():
                    events.put(("result", response.model_dump()))
        except Exception as exc:
            logger.warning("Streaming mailbox request failed: %s", type(exc).__name__)
            if not disconnected.is_set():
                events.put(("error", {"message": "The mailbox request could not finish. Please try again."}))
        finally:
            events.put(None)

    def generate():
        Thread(target=work, daemon=True).start()
        try:
            while True:
                try:
                    item = events.get(timeout=15)
                except Empty:
                    yield ": keepalive\n\n"
                    continue
                if item is None:
                    break
                kind, payload = item
                yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            disconnected.set()

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


@router.post("/index")
def trigger_rag_indexing(
    body: IndexRequest = IndexRequest(),
    user: User = Depends(get_user_for_rag),
    db: Session = Depends(get_db),
):
    """Run incremental indexing of user's emails into ChromaDB vector store."""
    try:
        results = default_pipeline.index_user_emails(
            db=db,
            user_id=user.id,
            batch_size=body.batch_size,
            force=body.full_reindex,
        )
        return {
            "status": "success",
            "user_id": str(user.id),
            "results": results,
            "message": (
                f"Indexing finished: {results['emails_indexed']} indexed, "
                f"{results['emails_skipped']} skipped, {results['chunks_created']} chunks stored."
            ),
        }
    except Exception as exc:
        logger.exception("Indexing error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(exc)}")


@router.get("/status")
def get_rag_system_status(
    user: User = Depends(get_user_for_rag),
    db: Session = Depends(get_db),
):
    """Return live status of ChromaDB vector collection, chunks, and pending approvals."""
    vector_count = default_vectorstore.count(user.id) if settings.vector_backend == "postgres" else default_vectorstore.count()

    total_chunks = db.scalar(
        select(func.count(EmailChunk.id))
        .join(Email, Email.id == EmailChunk.email_id)
        .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
        .where(GmailAccount.user_id == user.id)
    ) or 0

    total_embeddings_indexed = db.scalar(
        select(func.count(EmailEmbedding.id))
        .join(EmailChunk, EmailChunk.id == EmailEmbedding.chunk_id)
        .join(Email, Email.id == EmailChunk.email_id)
        .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
        .where(GmailAccount.user_id == user.id, EmailEmbedding.status == "INDEXED")
    ) or 0

    pending_approvals = db.scalar(
        select(func.count(EmailSendApproval.id))
        .where(EmailSendApproval.user_id == user.id, EmailSendApproval.status == "PENDING")
    ) or 0

    user_email_count = db.scalar(
        select(func.count(Email.id))
        .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
        .where(GmailAccount.user_id == user.id)
    ) or 0

    return {
        "status": "operational",
        "vector_store": {
            "collection_name": default_vectorstore.collection_name,
            "backend": settings.vector_backend,
            "total_vectors": vector_count,
        },
        "postgresql": {
            "user_emails": user_email_count,
            "total_chunks": total_chunks,
            "indexed_embeddings": total_embeddings_indexed,
            "indexed_emails": db.scalar(select(func.count(func.distinct(Email.id))).join(EmailChunk, EmailChunk.email_id == Email.id).join(EmailEmbedding, EmailEmbedding.chunk_id == EmailChunk.id).join(GmailAccount, GmailAccount.id == Email.gmail_account_id).where(GmailAccount.user_id == user.id, EmailEmbedding.status == "INDEXED")) or 0,
            "pending_emails": max(0, user_email_count - (db.scalar(select(func.count(func.distinct(Email.id))).join(EmailChunk, EmailChunk.email_id == Email.id).join(EmailEmbedding, EmailEmbedding.chunk_id == EmailChunk.id).join(GmailAccount, GmailAccount.id == Email.gmail_account_id).where(GmailAccount.user_id == user.id, EmailEmbedding.status == "INDEXED")) or 0)),
            "pending_approvals": pending_approvals,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/sessions")
def get_chat_sessions(
    limit: int = 20,
    user: User = Depends(get_user_for_rag),
    db: Session = Depends(get_db),
):
    """List recent conversation sessions for the current user."""
    return list_user_sessions(db, user.id, limit=limit)


@router.get("/sessions/{session_id}")
def get_session_details(
    session_id: str,
    user: User = Depends(get_user_for_rag),
    db: Session = Depends(get_db),
):
    """Retrieve full message history and citations for a specific chat session."""
    try:
        s_uuid = UUID(session_id.strip())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session UUID format.")

    session_rec = db.scalar(
        select(ChatSession).where(
            ChatSession.id == s_uuid,
            ChatSession.user_id == user.id,
        )
    )
    if not session_rec:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    messages = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.chat_session_id == s_uuid)
        .order_by(ChatMessage.created_at.asc())
    ).all()

    serialized_messages = []
    for msg in messages:
        # Load citation chunks linked to this message
        retrievals = db.scalars(
            select(ChatRetrieval)
            .where(ChatRetrieval.chat_message_id == msg.id)
            .order_by(ChatRetrieval.retrieval_rank.asc())
        ).all()

        cits = []
        for r in retrievals:
            chunk = r.chunk
            if chunk and chunk.email:
                cits.append({
                    "email_id": str(chunk.email.id),
                    "subject": chunk.email.subject or "",
                    "sender": chunk.email.from_email or "",
                    "date": chunk.email.received_at.isoformat() if chunk.email.received_at else "",
                    "snippet": chunk.chunk_text[:200],
                    "score": round(r.similarity_score or 0.0, 4),
                })

        serialized_messages.append({
            "id": str(msg.id),
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat() if msg.created_at else "",
            "citations": cits,
        })

    return {
        "id": str(session_rec.id),
        "title": session_rec.title or "Untitled Conversation",
        "created_at": session_rec.created_at.isoformat() if session_rec.created_at else "",
        "updated_at": session_rec.updated_at.isoformat() if session_rec.updated_at else "",
        "messages": serialized_messages,
    }


@router.delete("/sessions/{session_id}")
def delete_chat_session(
    session_id: str,
    user: User = Depends(get_user_for_rag),
    db: Session = Depends(get_db),
):
    """Delete a chat session and associated messages."""
    try:
        s_uuid = UUID(session_id.strip())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session UUID.")

    session_rec = db.scalar(
        select(ChatSession).where(
            ChatSession.id == s_uuid,
            ChatSession.user_id == user.id,
        )
    )
    if not session_rec:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    db.delete(session_rec)
    db.commit()
    return {"status": "deleted", "session_id": str(s_uuid)}


@router.get("/approvals")
def get_approvals(
    status_filter: Optional[str] = None,
    user: User = Depends(get_user_for_rag),
    db: Session = Depends(get_db),
):
    """List pending or historical email send approvals (Human-in-the-Loop)."""
    query = select(EmailSendApproval).where(EmailSendApproval.user_id == user.id)
    if status_filter:
        query = query.where(EmailSendApproval.status == status_filter.upper())
    query = query.order_by(desc(EmailSendApproval.created_at))

    approvals = db.scalars(query).all()
    return [
        {
            "id": str(a.id),
            "recipient_email": a.recipient_email,
            "subject": a.subject or "",
            "draft_body": a.draft_body,
            "status": a.status,
            "created_at": a.created_at.isoformat() if a.created_at else "",
            "approved_at": a.approved_at.isoformat() if a.approved_at else None,
            "sent_at": a.sent_at.isoformat() if a.sent_at else None,
        }
        for a in approvals
    ]


@router.post("/approvals/{approval_id}/approve", response_model=ApprovalActionResponse)
def approve_email_send(
    approval_id: str,
    user: User = Depends(get_user_for_rag),
    db: Session = Depends(get_db),
):
    """Explicit human approval to authorize sending drafted email."""
    try:
        a_uuid = UUID(approval_id.strip())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid approval UUID.")

    approval = db.scalar(
        select(EmailSendApproval).where(
            EmailSendApproval.id == a_uuid,
            EmailSendApproval.user_id == user.id,
        )
    )
    if not approval:
        raise HTTPException(status_code=404, detail="Approval request not found.")

    if approval.status != "PENDING":
        raise HTTPException(status_code=400, detail=f"Draft cannot be approved from status '{approval.status}'.")

    approval.status = "APPROVED"
    approval.approved_at = datetime.utcnow()
    db.commit()

    return ApprovalActionResponse(
        approval_id=str(approval.id),
        status="APPROVED",
        message=f"Draft for {approval.recipient_email} approved. Delivery is not configured; no email was sent.",
    )


@router.post("/approvals/{approval_id}/reject", response_model=ApprovalActionResponse)
def reject_email_send(
    approval_id: str,
    user: User = Depends(get_user_for_rag),
    db: Session = Depends(get_db),
):
    """Explicit human rejection to cancel drafted email."""
    try:
        a_uuid = UUID(approval_id.strip())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid approval UUID.")

    approval = db.scalar(
        select(EmailSendApproval).where(
            EmailSendApproval.id == a_uuid,
            EmailSendApproval.user_id == user.id,
        )
    )
    if not approval:
        raise HTTPException(status_code=404, detail="Approval request not found.")

    approval.status = "REJECTED"
    db.commit()

    return ApprovalActionResponse(
        approval_id=str(approval.id),
        status="REJECTED",
        message=f"Draft email to {approval.recipient_email} rejected.",
    )


@router.get("/sources/{email_id}")
def get_source_email_details(
    email_id: str,
    user: User = Depends(get_user_for_rag),
    db: Session = Depends(get_db),
):
    """Fetch full email content and metadata for citation inspector."""
    try:
        e_uuid = UUID(email_id.strip())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid email UUID.")

    email = db.scalar(
        select(Email)
        .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
        .where(Email.id == e_uuid, GmailAccount.user_id == user.id)
    )
    if not email:
        raise HTTPException(status_code=404, detail="Source email not found.")

    attachments = db.scalars(
        select(Attachment).where(Attachment.email_id == email.id)
    ).all()

    sender_name, sender_address = parseaddr(email.from_email or "")
    received_at = email.received_at
    if received_at and received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=timezone.utc)

    return {
        "id": str(email.id),
        "subject": email.subject or "",
        "from": email.from_email or "",
        "from_email": sender_address or email.from_email or "",
        "sender_name": email.sender_name or sender_name,
        "to": email.to_email or "",
        "cc": email.cc or "",
        "bcc": email.bcc or "",
        "date": received_at.isoformat() if received_at else None,
        "received_at": received_at.isoformat() if received_at else None,
        "snippet": email.snippet or "",
        "body": clean_email_text(email.body, email.body_html, remove_quotes=False, remove_signatures=False),
        "body_html": email.body_html or "",
        "is_read": email.is_read,
        "attachments": [
            {
                "id": str(a.id),
                "filename": a.filename,
                "mime_type": a.mime_type,
                "size_bytes": a.file_size_bytes,
                "extraction_status": a.extraction_status,
            }
            for a in attachments
        ],
    }

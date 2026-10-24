"""LangChain conversational memory integrated with PostgreSQL.

Implements:
- LangChain BaseChatMessageHistory backed by PostgreSQL chat_messages table
- Session lifecycle management (get/create/list)
- Citation audit logging in chat_retrievals table
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID, uuid4

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.models.email import Email, GmailAccount
from app.models.rag import ChatMessage, ChatRetrieval, ChatSession, EmailChunk
from app.rag.retrieval import RetrievedCitation

logger = logging.getLogger(__name__)


class PostgresChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat message history backed by PostgreSQL database."""

    def __init__(self, db: Session, session_id: UUID | str):
        self.db = db
        self.session_id = UUID(str(session_id))

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve historical messages as LangChain message instances."""
        db_messages = self.db.scalars(
            select(ChatMessage)
            .where(ChatMessage.chat_session_id == self.session_id)
            .order_by(ChatMessage.created_at.asc())
        ).all()

        lc_messages: List[BaseMessage] = []
        for msg in db_messages:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            else:
                lc_messages.append(HumanMessage(content=msg.content))
        return lc_messages

    def add_message(self, message: BaseMessage) -> None:
        """Persist a single LangChain message to PostgreSQL."""
        role = "user"
        if isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"

        content = message.content if isinstance(message.content, str) else str(message.content)

        db_msg = ChatMessage(
            chat_session_id=self.session_id,
            role=role,
            content=content,
        )
        self.db.add(db_msg)

        # Update session updated_at timestamp
        session = self.db.scalar(select(ChatSession).where(ChatSession.id == self.session_id))
        if session:
            session.updated_at = datetime.utcnow()

        self.db.commit()

    def clear(self) -> None:
        """Clear all messages from this session."""
        msgs = self.db.scalars(
            select(ChatMessage).where(ChatMessage.chat_session_id == self.session_id)
        ).all()
        for m in msgs:
            self.db.delete(m)
        self.db.commit()


def get_or_create_chat_session(
    db: Session,
    user_id: UUID | str,
    session_id: Optional[UUID | str] = None,
    title: Optional[str] = None,
) -> ChatSession:
    """Retrieve an existing user session or create a new one."""
    user_uuid = UUID(str(user_id))

    if session_id:
        try:
            s_uuid = UUID(str(session_id))
            existing = db.scalar(
                select(ChatSession).where(
                    ChatSession.id == s_uuid,
                    ChatSession.user_id == user_uuid,
                )
            )
            if existing:
                if title and not existing.title:
                    existing.title = title
                    db.commit()
                return existing
        except Exception as exc:
            logger.warning("Invalid session ID format %s: %s", session_id, exc)

    new_session = ChatSession(
        id=uuid4(),
        user_id=user_uuid,
        title=title or "New Conversation",
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


def record_retrieval_citations(
    db: Session,
    session_id: UUID | str,
    message_id: UUID | str,
    citations: List[RetrievedCitation],
) -> None:
    """Store only user-owned retrieved citations in the audit table.

    Retrieval results are expected to be scoped already, but this is a final
    persistence boundary.  Re-checking both the chat session owner and the
    email owner prevents a malformed citation or stale index record from
    linking another user's email into a session.
    """
    s_uuid = UUID(str(session_id))
    m_uuid = UUID(str(message_id))

    session = db.scalar(select(ChatSession).where(ChatSession.id == s_uuid))
    message = db.scalar(
        select(ChatMessage).where(
            ChatMessage.id == m_uuid,
            ChatMessage.chat_session_id == s_uuid,
        )
    )
    if session is None or message is None:
        logger.warning("Skipping retrieval audit for unknown session/message: %s/%s", s_uuid, m_uuid)
        return

    owned_chunk_query = (
        select(EmailChunk)
        .join(Email, Email.id == EmailChunk.email_id)
        .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
        .where(GmailAccount.user_id == session.user_id)
    )

    for rank, citation in enumerate(citations, 1):
        try:
            # Find the corresponding EmailChunk record in PostgreSQL
            email_uuid = UUID(str(citation.email_id))
            chunk = db.scalar(
                owned_chunk_query.where(
                    EmailChunk.email_id == email_uuid,
                    EmailChunk.chunk_index == citation.chunk_index,
                )
            )
            if not chunk:
                # Fallback to first chunk if specific index missing
                chunk = db.scalar(owned_chunk_query.where(EmailChunk.email_id == email_uuid))

            if chunk:
                retrieval_entry = ChatRetrieval(
                    chat_session_id=s_uuid,
                    chat_message_id=m_uuid,
                    chunk_id=chunk.id,
                    similarity_score=float(citation.score),
                    retrieval_rank=rank,
                )
                db.add(retrieval_entry)
        except Exception as exc:
            logger.debug("Could not record retrieval citation for %s: %s", citation.email_id, exc)

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("Failed to commit chat retrievals: %s", exc)


def list_user_sessions(
    db: Session,
    user_id: UUID | str,
    limit: int = 20,
) -> List[dict[str, Any]]:
    """List recent chat sessions for a user."""
    user_uuid = UUID(str(user_id))
    sessions = db.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == user_uuid)
        .order_by(desc(ChatSession.updated_at))
        .limit(limit)
    ).all()

    return [
        {
            "id": str(s.id),
            "title": s.title or "Untitled Conversation",
            "created_at": s.created_at.isoformat() if s.created_at else "",
            "updated_at": s.updated_at.isoformat() if s.updated_at else "",
            "message_count": len(s.messages),
        }
        for s in sessions
    ]

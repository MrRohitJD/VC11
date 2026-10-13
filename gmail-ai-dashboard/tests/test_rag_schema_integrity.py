"""Integrity and relationship tests for RAG tables, attachments, and chatbot models."""

from __future__ import annotations

import os
import sys
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gmail-ai-dashboard", "backend"))

from app.models.email import Base, Email, GmailAccount, User
import app.models.llm  # noqa: F401
import app.models.rag  # noqa: F401
from app.models.rag import (
    Attachment,
    ChatMessage,
    ChatRetrieval,
    ChatSession,
    EmailChunk,
    EmailEmbedding,
    EmailSendApproval,
)


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session, engine
    session.close()
    engine.dispose()


def test_all_sixteen_tables_exist(db):
    _, engine = db
    expected_tables = {
        # Core original tables (9)
        "users",
        "gmail_accounts",
        "emails",
        "labels",
        "email_labels",
        "ai_categories",
        "email_classifications",
        "sync_history",
        "llm_classification_jobs",
        # New RAG and chat tables (7)
        "email_chunks",
        "email_embeddings",
        "attachments",
        "chat_sessions",
        "chat_messages",
        "chat_retrievals",
        "email_send_approvals",
    }
    actual_tables = set(inspect(engine).get_table_names())
    assert expected_tables <= actual_tables


def test_email_chunk_and_embedding_lifecycle(db):
    session, _ = db
    user = User(google_id="user-rag-1", email="user-rag@example.com")
    session.add(user)
    session.flush()

    account = GmailAccount(user_id=user.id, email="account@example.com", access_token="token")
    session.add(account)
    session.flush()

    email = Email(gmail_account_id=account.id, gmail_message_id="msg-123", subject="Interview Offer")
    session.add(email)
    session.flush()

    chunk_0 = EmailChunk(
        email_id=email.id,
        chunk_index=0,
        chunk_text="Subject: Interview Offer\nDear Candidate...",
        start_index=0,
        end_index=35,
        chunk_strategy="paragraph",
    )
    chunk_1 = EmailChunk(
        email_id=email.id,
        chunk_index=1,
        chunk_text="Please join us on Monday...",
        start_index=36,
        end_index=65,
        chunk_strategy="paragraph",
    )
    session.add_all([chunk_0, chunk_1])
    session.flush()

    embedding_0 = EmailEmbedding(
        chunk_id=chunk_0.id,
        chroma_collection="gmail_email_chunks",
        chroma_record_id=f"rec-{chunk_0.id}",
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimensions=384,
        content_hash="abc123hash",
        status="INDEXED",
    )
    session.add(embedding_0)
    session.commit()

    # Verify duplicate chunk_index on same email violates uniqueness
    duplicate_chunk = EmailChunk(
        email_id=email.id,
        chunk_index=0,
        chunk_text="Conflict chunk",
    )
    session.add(duplicate_chunk)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_attachment_deduplication(db):
    session, _ = db
    user = User(google_id="user-att", email="att@example.com")
    session.add(user)
    session.flush()

    account = GmailAccount(user_id=user.id, email="att-acc@example.com", access_token="token")
    session.add(account)
    session.flush()

    email = Email(gmail_account_id=account.id, gmail_message_id="msg-att-1")
    session.add(email)
    session.flush()

    att1 = Attachment(
        email_id=email.id,
        gmail_attachment_id="att-id-001",
        filename="resume.pdf",
        mime_type="application/pdf",
        file_size_bytes=10240,
        extraction_status="COMPLETED",
        extracted_text="Rohit Jadhav Software Engineer Resume",
    )
    session.add(att1)
    session.commit()

    att2 = Attachment(
        email_id=email.id,
        gmail_attachment_id="att-id-001",
        filename="resume.pdf",
    )
    session.add(att2)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_chat_session_messages_and_retrievals(db):
    session, _ = db
    user = User(google_id="chat-user", email="chat@example.com")
    session.add(user)
    session.flush()

    account = GmailAccount(user_id=user.id, email="chat-acc@example.com", access_token="token")
    session.add(account)
    session.flush()

    email = Email(gmail_account_id=account.id, gmail_message_id="msg-chat-1")
    session.add(email)
    session.flush()

    chunk = EmailChunk(
        email_id=email.id,
        chunk_index=0,
        chunk_text="Infosys technical interview scheduled for 2 PM",
    )
    session.add(chunk)
    session.flush()

    chat_session = ChatSession(user_id=user.id, title="Interview Schedule Inquiries")
    session.add(chat_session)
    session.flush()

    user_msg = ChatMessage(chat_session_id=chat_session.id, role="user", content="When is my Infosys interview?")
    assistant_msg = ChatMessage(
        chat_session_id=chat_session.id,
        role="assistant",
        content="Your Infosys technical interview is scheduled for 2 PM.",
    )
    session.add_all([user_msg, assistant_msg])
    session.flush()

    retrieval = ChatRetrieval(
        chat_session_id=chat_session.id,
        chat_message_id=assistant_msg.id,
        chunk_id=chunk.id,
        similarity_score=0.945,
        retrieval_rank=1,
    )
    session.add(retrieval)
    session.commit()

    # Test cascade delete: deleting chat_session deletes messages and retrievals
    session.delete(chat_session)
    session.commit()

    assert session.query(ChatMessage).filter_by(chat_session_id=chat_session.id).count() == 0
    assert session.query(ChatRetrieval).filter_by(chat_session_id=chat_session.id).count() == 0
    # The email chunk itself remains intact!
    assert session.query(EmailChunk).filter_by(id=chunk.id).count() == 1


def test_email_send_approval(db):
    session, _ = db
    user = User(google_id="send-user", email="sender@example.com")
    session.add(user)
    session.flush()

    approval = EmailSendApproval(
        user_id=user.id,
        recipient_email="recruiter@company.com",
        subject="Re: Technical Interview Confirmation",
        draft_body="Thank you for the invitation, I confirm my availability for 2 PM.",
        status="PENDING",
    )
    session.add(approval)
    session.commit()

    assert approval.id is not None
    assert approval.status == "PENDING"


def test_envelope_metadata_chunking_and_embedding(db):
    session, _ = db
    user = User(google_id="user-envelope", email="user@example.com")
    session.add(user)
    session.flush()

    account = GmailAccount(user_id=user.id, email="user@example.com", access_token="tok")
    session.add(account)
    session.flush()

    from datetime import datetime, timezone
    email = Email(
        gmail_account_id=account.id,
        gmail_message_id="msg-env-100",
        from_email="recruiter@infosys.com",
        to_email="user@example.com",
        subject="Technical Interview Invitation",
        received_at=datetime(2026, 10, 5, 14, 0, tzinfo=timezone.utc),
        body="Dear Candidate, we invite you to interview for the Senior Full Stack role.",
    )
    session.add(email)
    session.flush()

    # Envelope Header Chunk (Index 0)
    envelope_text = (
        f"[EMAIL ENVELOPE]\n"
        f"Subject: {email.subject}\n"
        f"From: {email.from_email}\n"
        f"To: {email.to_email}\n"
        f"Date: {email.received_at.isoformat()}\n"
        f"Category: JOB\n"
        f"Summary / Preview: {email.body[:100]}"
    )
    chunk_0 = EmailChunk(
        email_id=email.id,
        chunk_index=0,
        chunk_text=envelope_text,
        chunk_strategy="envelope_header",
    )

    # Contextual Body Chunk (Index 1)
    body_text = (
        f"Subject: {email.subject}\n"
        f"From: {email.from_email}\n"
        f"To: {email.to_email}\n"
        f"Date: {email.received_at.isoformat()}\n"
        f"Category: JOB\n\n"
        f"Content:\n{email.body}"
    )
    chunk_1 = EmailChunk(
        email_id=email.id,
        chunk_index=1,
        chunk_text=body_text,
        chunk_strategy="contextual_body",
    )
    session.add_all([chunk_0, chunk_1])
    session.flush()

    assert "To: user@example.com" in chunk_0.chunk_text
    assert "From: recruiter@infosys.com" in chunk_0.chunk_text
    assert "Subject: Technical Interview Invitation" in chunk_0.chunk_text
    assert "2026-10-05" in chunk_0.chunk_text
    assert chunk_0.chunk_strategy == "envelope_header"
    assert chunk_1.chunk_strategy == "contextual_body"
    assert len(email.chunks) == 2

"""Tests for hybrid email chunking, subject formatting, idempotency, and user-scoped ChromaDB retrieval."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from uuid import uuid4

import chromadb
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gmail-ai-dashboard", "backend"))

from app.models.email import Base, Email, GmailAccount, User
import app.models.llm  # noqa: F401
import app.models.rag  # noqa: F401
from app.models.rag import EmailChunk, EmailEmbedding
from app.services.email_chunker import (
    clean_email_body,
    create_email_chunks,
    format_embedding_text,
)
from app.services.rag_indexer import RAGIndexerService


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture()
def rag_service():
    # Use in-memory Chroma client for fast, isolated test execution
    client = chromadb.EphemeralClient()
    return RAGIndexerService(chroma_client=client, collection_name=f"test_gmail_email_chunks_{uuid4().hex}")


# ---------------------------------------------------------------------------
# 1. Chunking Unit Tests
# ---------------------------------------------------------------------------

def test_short_email_remains_one_chunk():
    """Short emails must stay intact as exactly one chunk with full_message strategy."""
    subject = "Quick question"
    body = "Hi Rohit,\nAre we still meeting at 3 PM today?\nThanks!"
    chunks = create_email_chunks(subject=subject, body=body, chunk_size=2000, chunk_overlap=250)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_index == 0
    assert chunk.chunk_strategy == "full_message"
    assert "Are we still meeting at 3 PM today?" in chunk.chunk_text
    assert chunk.start_index == 0
    assert chunk.end_index == len(chunk.chunk_text)


def test_long_email_produces_multiple_overlapping_chunks():
    """Emails exceeding chunk_size must be split into multiple overlapping chunks."""
    subject = "Comprehensive Architecture Review"
    paragraphs = [
        f"Section {i}: This is detailed engineering discussion paragraph number {i}. "
        f"It contains in-depth analysis of PostgreSQL schemas, Alembic migrations, "
        f"and vector retrieval strategies with ChromaDB. " * 5
        for i in range(1, 15)
    ]
    body = "\n\n".join(paragraphs)
    assert len(body) > 3000

    chunks = create_email_chunks(subject=subject, body=body, chunk_size=1000, chunk_overlap=150)

    assert len(chunks) > 1
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.chunk_strategy == "recursive_character"
        assert len(c.chunk_text) <= 1200  # Split roughly within threshold


def test_subject_information_included_in_embedding_text():
    """Embedding text must include the email subject for semantic discovery."""
    subject = "Contract Renewal 2026"
    body = "Please sign the attached agreement."
    chunks = create_email_chunks(subject=subject, body=body)

    assert len(chunks) == 1
    embedding_text = chunks[0].embedding_text
    assert embedding_text.startswith("Subject: Contract Renewal 2026\n\nEmail content:\n")
    assert "Please sign the attached agreement." in embedding_text


def test_cleaning_strips_html_quotes_and_signatures():
    """Email body normalizer removes HTML, quoted replies, and standard signatures."""
    html_body = "<div>Hello!</div><p>Here is the update.</p>"
    cleaned_html = clean_email_body(None, body_html=html_body)
    assert "Hello!" in cleaned_html
    assert "Here is the update." in cleaned_html
    assert "<div" not in cleaned_html

    email_with_quotes = (
        "Sounds good, let us proceed.\n\n"
        "On Oct 12, 2026, at 10:00 AM, Alice wrote:\n"
        "> Shall we proceed with the migration?\n"
        "> Let me know your thoughts."
    )
    cleaned_quotes = clean_email_body(email_with_quotes)
    assert cleaned_quotes == "Sounds good, let us proceed."

    email_with_signature = (
        "Here are the deliverables.\n\n"
        "-- \n"
        "Rohit Jadhav\n"
        "Senior Software Engineer"
    )
    cleaned_sig = clean_email_body(email_with_signature)
    assert cleaned_sig == "Here are the deliverables."


# ---------------------------------------------------------------------------
# 2. Integration Tests: PostgreSQL & ChromaDB
# ---------------------------------------------------------------------------

def test_repeated_indexing_is_idempotent_no_duplicates(db_session, rag_service):
    """Re-indexing unchanged email must not create duplicate PostgreSQL chunks or ChromaDB vectors."""
    user = User(google_id="u-idem-1", email="user@test.com")
    db_session.add(user)
    db_session.flush()

    account = GmailAccount(user_id=user.id, email="user@test.com", access_token="tok")
    db_session.add(account)
    db_session.flush()

    email = Email(
        gmail_account_id=account.id,
        gmail_message_id="msg-idem-1",
        from_email="billing@aws.com",
        to_email="user@test.com",
        subject="Monthly Invoice",
        received_at=datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc),
        body="Your total bill for September is $45.00.",
    )
    db_session.add(email)
    db_session.commit()

    # First indexing
    count_1 = rag_service.index_email(db=db_session, email_id=email.id, user_id=user.id)
    assert count_1 == 1
    assert db_session.query(EmailChunk).filter_by(email_id=email.id).count() == 1
    assert db_session.query(EmailEmbedding).count() == 1
    assert rag_service.collection.count() == 1

    # Second indexing (identical content) -> Must be a NO-OP
    count_2 = rag_service.index_email(db=db_session, email_id=email.id, user_id=user.id)
    assert count_2 == 1
    assert db_session.query(EmailChunk).filter_by(email_id=email.id).count() == 1
    assert db_session.query(EmailEmbedding).count() == 1
    assert rag_service.collection.count() == 1


def test_updated_email_content_triggers_reindexing(db_session, rag_service):
    """When an email body changes, previous vectors are replaced without leaving orphans."""
    user = User(google_id="u-upd-1", email="user@test.com")
    db_session.add(user)
    db_session.flush()

    account = GmailAccount(user_id=user.id, email="user@test.com", access_token="tok")
    db_session.add(account)
    db_session.flush()

    email = Email(
        gmail_account_id=account.id,
        gmail_message_id="msg-upd-1",
        from_email="hr@company.com",
        to_email="user@test.com",
        subject="Interview Schedule",
        received_at=datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc),
        body="Initial schedule: Oct 10 at 10 AM.",
    )
    db_session.add(email)
    db_session.commit()

    rag_service.index_email(db=db_session, email_id=email.id, user_id=user.id)
    assert rag_service.collection.count() == 1

    # Update email body
    email.body = "Updated schedule: Oct 15 at 2 PM. Please confirm availability."
    db_session.commit()

    # Re-indexing
    rag_service.index_email(db=db_session, email_id=email.id, user_id=user.id)
    assert rag_service.collection.count() == 1  # Replaced, not duplicated!

    # Verify retrieved vector contains updated text
    results = rag_service.search_rag(query="when is the updated interview?", user_id=user.id)
    assert len(results) == 1
    assert "Oct 15 at 2 PM" in results[0]["document"]


def test_deleted_emails_removed_from_retrieval(db_session, rag_service):
    """Deleted emails must be removed from ChromaDB so they never appear in search."""
    user = User(google_id="u-del-1", email="user@test.com")
    db_session.add(user)
    db_session.flush()

    account = GmailAccount(user_id=user.id, email="user@test.com", access_token="tok")
    db_session.add(account)
    db_session.flush()

    email = Email(
        gmail_account_id=account.id,
        gmail_message_id="msg-del-1",
        from_email="newsletter@tech.com",
        subject="Weekly Digest",
        body="Tech digest article content...",
    )
    db_session.add(email)
    db_session.commit()

    rag_service.index_email(db=db_session, email_id=email.id, user_id=user.id)
    assert rag_service.collection.count() == 1

    # Delete email from RAG
    rag_service.delete_email_from_rag(db=db_session, email_id=email.id)
    assert rag_service.collection.count() == 0

    results = rag_service.search_rag(query="Tech digest", user_id=user.id)
    assert len(results) == 0


def test_user_scoped_semantic_search_isolation(db_session, rag_service):
    """User A must never retrieve User B's email chunks in search results."""
    user_a = User(google_id="user-a", email="a@company.com")
    user_b = User(google_id="user-b", email="b@company.com")
    db_session.add_all([user_a, user_b])
    db_session.flush()

    acc_a = GmailAccount(user_id=user_a.id, email="a@company.com", access_token="tok-a")
    acc_b = GmailAccount(user_id=user_b.id, email="b@company.com", access_token="tok-b")
    db_session.add_all([acc_a, acc_b])
    db_session.flush()

    email_a = Email(
        gmail_account_id=acc_a.id,
        gmail_message_id="msg-a-1",
        subject="Secret Bonus Allocation",
        body="Confidential bonus details for employee A.",
    )
    email_b = Email(
        gmail_account_id=acc_b.id,
        gmail_message_id="msg-b-1",
        subject="Bonus Information",
        body="Confidential bonus details for employee B.",
    )
    db_session.add_all([email_a, email_b])
    db_session.commit()

    rag_service.index_email(db=db_session, email_id=email_a.id, user_id=user_a.id)
    rag_service.index_email(db=db_session, email_id=email_b.id, user_id=user_b.id)
    assert rag_service.collection.count() == 2

    # Query as User A
    results_for_a = rag_service.search_rag(query="bonus details", user_id=user_a.id)
    assert len(results_for_a) == 1
    assert results_for_a[0]["user_id"] == str(user_a.id)
    assert results_for_a[0]["email_id"] == str(email_a.id)

    # Query as User B
    results_for_b = rag_service.search_rag(query="bonus details", user_id=user_b.id)
    assert len(results_for_b) == 1
    assert results_for_b[0]["user_id"] == str(user_b.id)
    assert results_for_b[0]["email_id"] == str(email_b.id)

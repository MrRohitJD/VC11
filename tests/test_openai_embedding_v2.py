"""Tests for OpenAI text-embedding-3-large (v2) migration, 3072 dimensions, duplicate prevention, and user isolation."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gmail-ai-dashboard", "backend"))

from app.models.email import Base, Email, GmailAccount, User
import app.models.llm  # noqa: F401
import app.models.rag  # noqa: F401
from app.models.rag import EmailChunk, EmailEmbedding
from app.services.embedding_provider import (
    EmbeddingProvider,
    OPENAI_LARGE_DIMENSIONS,
    OPENAI_LARGE_MODEL,
)
from app.services.rag_indexer import RAGIndexerService
from app.services.rag_migration_service import MigrationProgress, RAGMigrationService


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
def mock_embedder():
    return EmbeddingProvider(mock_mode=True)


# ---------------------------------------------------------------------------
# 1. Embedding Dimension Verification Tests
# ---------------------------------------------------------------------------

def test_openai_embedding_dimensions_3072(mock_embedder):
    """Verify that query and document embeddings produce exactly 3072 dimensions."""
    query_vec = mock_embedder.embed_query("Technical interview schedule")
    assert len(query_vec) == 3072

    doc_vectors = mock_embedder.embed_documents([
        "Subject: Interview Offer\n\nEmail content:\nDetails about Monday.",
        "Subject: Billing Update\n\nEmail content:\nInvoice amount is $100.",
    ])
    assert len(doc_vectors) == 2
    assert len(doc_vectors[0]) == 3072
    assert len(doc_vectors[1]) == 3072


def test_postgres_migration_is_resumable_and_preserves_metadata(db_session, mock_embedder):
    user = User(google_id="pg-migration", email="migration@example.com")
    db_session.add(user)
    db_session.flush()
    account = GmailAccount(user_id=user.id, email=user.email, access_token="test")
    db_session.add(account)
    db_session.flush()
    email = Email(gmail_account_id=account.id, gmail_message_id="pg-migration",
                  subject="Quarterly report", body="Revenue increased this quarter.",
                  from_email="finance@example.com", to_email=user.email)
    db_session.add(email)
    db_session.flush()
    chunk = EmailChunk(email_id=email.id, chunk_index=0,
                       chunk_text=email.body, content_hash="migration-content")
    db_session.add(chunk)
    db_session.commit()
    service = RAGMigrationService(embedding_provider=mock_embedder)
    progress = []
    first = service.migrate_all_chunks(db_session, batch_size=1, progress_callback=progress.append)
    assert first.migrated == 1 and first.failed == 0
    assert progress[-1].percent_complete == 100
    stored = db_session.scalar(select(EmailEmbedding))
    assert stored.vector_collection == "postgres_halfvec_3072"
    assert stored.vector_record_id == f"pgvector_{chunk.id}"
    assert stored.email_subject == email.subject
    assert stored.sender_email == email.from_email
    assert stored.recipient_emails == [user.email]
    assert len(stored.embedding) == 3072
    second = service.migrate_all_chunks(db_session)
    assert second.already_indexed == 1 and second.migrated == 0
    assert db_session.query(EmailEmbedding).count() == 1

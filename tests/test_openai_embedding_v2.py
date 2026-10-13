"""Tests for OpenAI text-embedding-3-large (v2) migration, 3072 dimensions, duplicate prevention, and user isolation."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from uuid import uuid4

import chromadb
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
def chroma_client():
    return chromadb.EphemeralClient()


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


# ---------------------------------------------------------------------------
# 2. Separate Collections Preservation (Zero Vector Mixing)
# ---------------------------------------------------------------------------

def test_separate_chroma_collections_preserved(chroma_client, mock_embedder):
    """Legacy v1 and new v2 collections must coexist separately in ChromaDB."""
    v1_name = "gmail_email_chunks"
    v2_name = "gmail_email_chunks_v2"

    coll_v1 = chroma_client.create_collection(
        name=v1_name,
        metadata={"hnsw:space": "cosine", "dimensions": 384},
    )
    coll_v1.add(
        ids=["chunk_v1_legacy"],
        documents=["Legacy document with 384 dimensions"],
        metadatas=[{"email_id": "legacy-id"}],
    )

    coll_v2 = chroma_client.create_collection(
        name=v2_name,
        metadata={"hnsw:space": "cosine", "dimensions": 3072},
    )
    doc_text = "New document with 3072 dimensions"
    coll_v2.add(
        ids=["chunk_v2_new"],
        embeddings=[mock_embedder.embed_query(doc_text)],
        documents=[doc_text],
        metadatas=[{"email_id": "new-id"}],
    )

    assert coll_v1.count() == 1
    assert coll_v2.count() == 1
    assert coll_v1.name == "gmail_email_chunks"
    assert coll_v2.name == "gmail_email_chunks_v2"


# ---------------------------------------------------------------------------
# 3. Resumable Batch Migration Tests
# ---------------------------------------------------------------------------

def test_resumable_batch_migration_and_progress(db_session, chroma_client, mock_embedder):
    """Migration must process chunks in batches, report progress, and resume without re-work."""
    user = User(google_id="u-mig-1", email="mig@test.com")
    db_session.add(user)
    db_session.flush()

    account = GmailAccount(user_id=user.id, email="mig@test.com", access_token="tok")
    db_session.add(account)
    db_session.flush()

    email_1 = Email(gmail_account_id=account.id, gmail_message_id="m-1", subject="Job Offer", body="Offer details...")
    email_2 = Email(gmail_account_id=account.id, gmail_message_id="m-2", subject="Project Update", body="Project status...")
    db_session.add_all([email_1, email_2])
    db_session.flush()

    # Create 3 chunks in PostgreSQL
    chunk_1 = EmailChunk(email_id=email_1.id, chunk_index=0, chunk_text="Offer details...", content_hash="hash-1")
    chunk_2 = EmailChunk(email_id=email_2.id, chunk_index=0, chunk_text="Project status part 1...", content_hash="hash-2")
    chunk_3 = EmailChunk(email_id=email_2.id, chunk_index=1, chunk_text="Project status part 2...", content_hash="hash-3")
    db_session.add_all([chunk_1, chunk_2, chunk_3])
    db_session.commit()

    migration_service = RAGMigrationService(
        chroma_client=chroma_client,
        embedding_provider=mock_embedder,
        target_collection_name="test_gmail_email_chunks_v2_mig",
    )

    progress_reports: list[MigrationProgress] = []

    def on_progress(p: MigrationProgress):
        progress_reports.append(p)

    # First migration run: processes all 3 chunks in batches of 2
    summary_1 = migration_service.migrate_all_chunks(db=db_session, batch_size=2, progress_callback=on_progress)
    assert summary_1.total_chunks == 3
    assert summary_1.migrated == 3
    assert summary_1.already_indexed == 0
    assert len(progress_reports) >= 2

    # Verify PostgreSQL registry entries
    embeddings_in_db = db_session.scalars(
        select(EmailEmbedding).where(EmailEmbedding.chroma_collection == "test_gmail_email_chunks_v2_mig")
    ).all()
    assert len(embeddings_in_db) == 3
    for emb in embeddings_in_db:
        assert emb.embedding_model == OPENAI_LARGE_MODEL
        assert emb.embedding_dimensions == OPENAI_LARGE_DIMENSIONS
        assert emb.status == "INDEXED"

    # Second migration run: must resume and skip all 3 already-indexed chunks!
    summary_2 = migration_service.migrate_all_chunks(db=db_session, batch_size=2)
    assert summary_2.total_chunks == 3
    assert summary_2.already_indexed == 3
    assert summary_2.migrated == 0


# ---------------------------------------------------------------------------
# 4. Duplicate Prevention & Idempotency in v2
# ---------------------------------------------------------------------------

def test_duplicate_prevention_and_idempotency_v2(db_session, chroma_client, mock_embedder):
    """Re-indexing an email multiple times into v2 collection must not produce duplicate vectors."""
    user = User(google_id="u-idem-v2", email="idem@test.com")
    db_session.add(user)
    db_session.flush()

    account = GmailAccount(user_id=user.id, email="idem@test.com", access_token="tok")
    db_session.add(account)
    db_session.flush()

    email = Email(
        gmail_account_id=account.id,
        gmail_message_id="msg-idem-v2",
        subject="Important Notice",
        body="This is an important system notification.",
    )
    db_session.add(email)
    db_session.commit()

    service = RAGIndexerService(
        chroma_client=chroma_client,
        collection_name="test_gmail_email_chunks_v2_idem",
        embedding_provider=mock_embedder,
    )

    # First index
    count_1 = service.index_email(db=db_session, email_id=email.id, user_id=user.id)
    assert count_1 == 1
    assert service.collection.count() == 1

    # Second index (identical) -> Idempotent NO-OP
    count_2 = service.index_email(db=db_session, email_id=email.id, user_id=user.id)
    assert count_2 == 1
    assert service.collection.count() == 1


# ---------------------------------------------------------------------------
# 5. User-Scoped Semantic Search in v2
# ---------------------------------------------------------------------------

def test_v2_user_scoped_semantic_search_isolation(db_session, chroma_client, mock_embedder):
    """User A must never retrieve User B's chunks in v2 semantic search."""
    user_a = User(google_id="usr-a", email="a@corp.com")
    user_b = User(google_id="usr-b", email="b@corp.com")
    db_session.add_all([user_a, user_b])
    db_session.flush()

    acc_a = GmailAccount(user_id=user_a.id, email="a@corp.com", access_token="tok-a")
    acc_b = GmailAccount(user_id=user_b.id, email="b@corp.com", access_token="tok-b")
    db_session.add_all([acc_a, acc_b])
    db_session.flush()

    email_a = Email(gmail_account_id=acc_a.id, gmail_message_id="m-a", subject="Q4 Roadmap", body="Confidential Q4 plans for User A.")
    email_b = Email(gmail_account_id=acc_b.id, gmail_message_id="m-b", subject="Q4 Budget", body="Confidential Q4 plans for User B.")
    db_session.add_all([email_a, email_b])
    db_session.commit()

    service = RAGIndexerService(
        chroma_client=chroma_client,
        collection_name="test_gmail_email_chunks_v2_isolation",
        embedding_provider=mock_embedder,
    )

    service.index_email(db=db_session, email_id=email_a.id, user_id=user_a.id)
    service.index_email(db=db_session, email_id=email_b.id, user_id=user_b.id)
    assert service.collection.count() == 2

    # Query as User A
    results_a = service.search_rag(query="Q4 plans", user_id=user_a.id)
    assert len(results_a) == 1
    assert results_a[0]["user_id"] == str(user_a.id)
    assert results_a[0]["subject"] == "Q4 Roadmap"

    # Query as User B
    results_b = service.search_rag(query="Q4 plans", user_id=user_b.id)
    assert len(results_b) == 1
    assert results_b[0]["user_id"] == str(user_b.id)
    assert results_b[0]["subject"] == "Q4 Budget"


# ---------------------------------------------------------------------------
# 6. Side-by-Side Retrieval Comparison Test
# ---------------------------------------------------------------------------

def test_side_by_side_retrieval_comparison(db_session, chroma_client, mock_embedder):
    """Compare retrieval output between v1 and v2 collections."""
    migration_service = RAGMigrationService(
        chroma_client=chroma_client,
        embedding_provider=mock_embedder,
        target_collection_name="test_gmail_email_chunks_v2_comp",
    )

    # Seed v2
    user = User(google_id="u-comp", email="comp@test.com")
    db_session.add(user)
    db_session.flush()

    account = GmailAccount(user_id=user.id, email="comp@test.com", access_token="tok")
    db_session.add(account)
    db_session.flush()

    email = Email(
        gmail_account_id=account.id,
        gmail_message_id="m-comp",
        subject="Python Conference Invitation",
        body="We invite you to keynote at PyCon.",
    )
    db_session.add(email)
    db_session.commit()

    indexer = RAGIndexerService(
        chroma_client=chroma_client,
        collection_name="test_gmail_email_chunks_v2_comp",
        embedding_provider=mock_embedder,
    )
    indexer.index_email(db=db_session, email_id=email.id, user_id=user.id)

    comparison = migration_service.compare_retrieval(
        query="PyCon keynote",
        user_id=user.id,
        top_k=3,
        v1_collection_name="nonexistent_v1",
    )

    assert comparison.v2_collection == "test_gmail_email_chunks_v2_comp"
    assert len(comparison.v2_results) == 1
    assert comparison.v2_results[0]["subject"] == "Python Conference Invitation"
    assert comparison.v2_latency_ms >= 0.0

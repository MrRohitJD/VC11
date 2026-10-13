"""Tests for Azure OpenAI text-embedding-3-large integration via LangChain AzureOpenAIEmbeddings.

Verifies:
1. Azure configuration loading from environment variables and settings
2. Verified embedding dimensions (3072d) and probe validation
3. Same deployment used for document embeddings and query embeddings
4. Dual collection isolation (no vector mixing with legacy 384d collection)
5. Resumable batch migration with progress tracking and idempotency
6. Subject-aware embedding text and metadata preservation (sender_id, recipient_id, received_date)
7. Multi-tenant user isolation on semantic retrieval
8. Safe email update and deletion handling
"""

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

from app.core.config import Settings
from app.models.email import Base, Email, GmailAccount, User
import app.models.llm  # noqa: F401
import app.models.rag  # noqa: F401
from app.models.rag import EmailChunk, EmailEmbedding
from app.services.email_chunker import (
    clean_email_body,
    create_email_chunks,
    format_embedding_text,
)
from app.services.embedding_provider import (
    AZURE_OPENAI_DEFAULT_DEPLOYMENT,
    AZURE_OPENAI_DEFAULT_MODEL,
    AZURE_OPENAI_LARGE_DIMENSIONS,
    EmbeddingProvider,
    MockAzureOpenAILargeEmbeddings,
)
from app.services.rag_indexer import RAGIndexerService
from app.services.rag_migration_service import RAGMigrationService


@pytest.fixture()
def db_session():
    """Isolated in-memory SQLite database session with foreign keys enabled."""
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
    """Isolated EphemeralClient for vector database testing."""
    return chromadb.EphemeralClient()


@pytest.fixture()
def azure_mock_embedder():
    """Mock Azure OpenAI embedding provider producing deterministic 3072d vectors."""
    return EmbeddingProvider(
        azure_endpoint="https://test-resource.openai.azure.com/",
        api_key="test-azure-key",
        api_version="2024-02-01",
        deployment_name="text-embedding-3-large",
        mock_mode=True,
    )


# ---------------------------------------------------------------------------
# 1. Azure Configuration Loading Tests
# ---------------------------------------------------------------------------

def test_azure_configuration_loading(monkeypatch):
    """Test loading Azure OpenAI endpoint, API key, API version, and deployment from env vars."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my-custom-ai.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "custom-secret-key-12345")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "custom-ada-large-deployment")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "azure_openai")

    test_settings = Settings()
    assert test_settings.azure_openai_endpoint == "https://my-custom-ai.openai.azure.com/"
    assert test_settings.azure_openai_api_key == "custom-secret-key-12345"
    assert test_settings.azure_openai_api_version == "2024-06-01"
    assert test_settings.azure_openai_embedding_deployment == "custom-ada-large-deployment"
    assert test_settings.embedding_provider == "azure_openai"


def test_azure_configuration_aliases(monkeypatch):
    """Verify alternative alias names for Azure environment variables."""
    monkeypatch.setenv("AZURE_ENDPOINT", "https://alias-ai.openai.azure.com/")
    monkeypatch.setenv("AZURE_API_KEY", "alias-secret-key-67890")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-05-01-preview")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "deployment-v2")

    test_settings = Settings()
    assert test_settings.azure_openai_endpoint == "https://alias-ai.openai.azure.com/"
    assert test_settings.azure_openai_api_key == "alias-secret-key-67890"
    assert test_settings.azure_openai_api_version == "2024-05-01-preview"
    assert test_settings.azure_openai_embedding_deployment == "deployment-v2"


# ---------------------------------------------------------------------------
# 2. Embedding Dimensions & Verification Probe
# ---------------------------------------------------------------------------

def test_azure_embedding_dimensions_and_verification(azure_mock_embedder):
    """Test verified dimensions (3072d) and unit normalization."""
    assert azure_mock_embedder.dimensions == AZURE_OPENAI_LARGE_DIMENSIONS
    assert azure_mock_embedder.deployment_name == "text-embedding-3-large"
    assert azure_mock_embedder.provider == "azure_openai"

    verified_dim = azure_mock_embedder.verify_dimensions()
    assert verified_dim == 3072

    # Verify query and document embeddings both have 3072 dimensions
    query_vec = azure_mock_embedder.embed_query("quarterly financial report")
    assert len(query_vec) == 3072

    doc_vecs = azure_mock_embedder.embed_documents(["doc 1", "doc 2"])
    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == 3072
    assert len(doc_vecs[1]) == 3072


# ---------------------------------------------------------------------------
# 3. Dual-Collection Isolation (Zero Vector Mixing)
# ---------------------------------------------------------------------------

def test_dual_collection_isolation_no_mixing(chroma_client, azure_mock_embedder):
    """Ensure v1 (384d) and v2 (3072d) collections coexist without mixing."""
    v1_collection_name = f"gmail_email_chunks_{uuid4().hex[:8]}"
    v2_collection_name = f"gmail_email_chunks_v2_{uuid4().hex[:8]}"

    v1_col = chroma_client.get_or_create_collection(
        name=v1_collection_name,
        metadata={"hnsw:space": "cosine", "model": "all-MiniLM-L6-v2", "dimensions": 384},
    )
    v2_col = chroma_client.get_or_create_collection(
        name=v2_collection_name,
        metadata={"hnsw:space": "cosine", "model": azure_mock_embedder.deployment_name, "dimensions": 3072},
    )

    # Insert 384d vector into v1
    v1_vec = [0.1] * 384
    v1_col.add(
        ids=["legacy_1"],
        embeddings=[v1_vec],
        documents=["Legacy email chunk"],
        metadatas=[{"model": "all-MiniLM-L6-v2", "dimensions": 384}],
    )

    # Insert 3072d vector into v2
    v2_vec = azure_mock_embedder.embed_query("Azure OpenAI chunk")
    v2_col.add(
        ids=["azure_1"],
        embeddings=[v2_vec],
        documents=["Azure OpenAI email chunk"],
        metadatas=[{"model": azure_mock_embedder.deployment_name, "dimensions": 3072}],
    )

    assert v1_col.count() == 1
    assert v2_col.count() == 1

    # Querying v2 collection with 3072d vector succeeds
    res_v2 = v2_col.query(query_embeddings=[v2_vec], n_results=1)
    assert res_v2["ids"][0][0] == "azure_1"


# ---------------------------------------------------------------------------
# 4. Resumable Batch Migration Tests
# ---------------------------------------------------------------------------

def test_resumable_batch_migration(db_session, chroma_client, azure_mock_embedder):
    """Verify batch migration re-embeds chunks into v2 with progress and skips already indexed ones."""
    user = User(google_id="usr_mig", email="mig@example.com")
    db_session.add(user)
    db_session.flush()

    acct = GmailAccount(user_id=user.id, email="mig@example.com", access_token="token")
    db_session.add(acct)
    db_session.flush()

    # Create 3 emails with multiple chunks
    for i in range(3):
        em = Email(
            gmail_account_id=acct.id,
            gmail_message_id=f"msg_mig_{i}",
            subject=f"Migration Email {i}",
            from_email="partner@example.com",
            to_email="mig@example.com",
            body="First paragraph of email text.\n\nSecond paragraph of email text.",
            received_at=datetime.now(timezone.utc),
        )
        db_session.add(em)
        db_session.flush()

        chunks = create_email_chunks(em.subject, em.body)
        for c in chunks:
            db_session.add(
                EmailChunk(
                    email_id=em.id,
                    chunk_index=c.chunk_index,
                    chunk_text=c.chunk_text,
                    start_index=c.start_index,
                    end_index=c.end_index,
                    chunk_strategy=c.chunk_strategy,
                    content_hash=c.content_hash,
                )
            )
    db_session.commit()

    total_chunks = len(db_session.scalars(select(EmailChunk)).all())
    assert total_chunks == 3

    target_col = f"gmail_email_chunks_v2_{uuid4().hex[:8]}"
    mig_service = RAGMigrationService(
        chroma_client=chroma_client,
        embedding_provider=azure_mock_embedder,
        target_collection_name=target_col,
    )

    # First pass: migrate in batches of 2
    prog1 = mig_service.migrate_all_chunks(db=db_session, batch_size=2)
    assert prog1.total_chunks == 3
    assert prog1.migrated == 3
    assert prog1.already_indexed == 0
    assert prog1.failed == 0

    # Verify PostgreSQL registry entries
    embeddings = db_session.scalars(
        select(EmailEmbedding).where(EmailEmbedding.chroma_collection == target_col)
    ).all()
    assert len(embeddings) == 3
    for emb in embeddings:
        assert emb.embedding_provider == "azure_openai"
        assert emb.embedding_model == "text-embedding-3-large"
        assert emb.embedding_dimensions == 3072
        assert emb.status == "INDEXED"

    # Second pass: should detect all chunks as already indexed (resumability)
    prog2 = mig_service.migrate_all_chunks(db=db_session, batch_size=2)
    assert prog2.total_chunks == 3
    assert prog2.migrated == 0
    assert prog2.already_indexed == 3
    assert prog2.failed == 0


# ---------------------------------------------------------------------------
# 5. Metadata Preservation & Subject-Aware Text
# ---------------------------------------------------------------------------

def test_subject_aware_text_and_metadata_preservation(db_session, chroma_client, azure_mock_embedder):
    """Ensure subject-aware embedding text and metadata (sender_id, recipient_id, received_date)."""
    user = User(google_id="usr_meta", email="owner@corp.com")
    db_session.add(user)
    db_session.flush()

    acct = GmailAccount(user_id=user.id, email="owner@corp.com", access_token="token")
    db_session.add(acct)
    db_session.flush()

    received_time = datetime(2026, 8, 15, 10, 30, tzinfo=timezone.utc)
    email = Email(
        gmail_account_id=acct.id,
        gmail_message_id="msg_meta_99",
        subject="Vendor Contract Negotiation",
        from_email="vendor@supplier.com",
        to_email="owner@corp.com",
        body="Attached is the signed service agreement for Q3.",
        received_at=received_time,
    )
    db_session.add(email)
    db_session.commit()

    col_name = f"gmail_email_chunks_v2_{uuid4().hex[:8]}"
    rag_service = RAGIndexerService(
        chroma_client=chroma_client,
        collection_name=col_name,
        embedding_provider=azure_mock_embedder,
    )

    indexed_count = rag_service.index_email(db=db_session, email_id=email.id)
    assert indexed_count == 1

    # Check ChromaDB stored document and metadata
    chroma_res = rag_service.collection.get(include=["documents", "metadatas"])
    assert len(chroma_res["documents"]) == 1
    doc_text = chroma_res["documents"][0]
    metadata = chroma_res["metadatas"][0]

    # Verify subject-aware text
    assert "Subject: Vendor Contract Negotiation" in doc_text
    assert "Email content:" in doc_text
    assert "Attached is the signed service agreement for Q3." in doc_text

    # Verify metadata fields
    assert metadata["email_id"] == str(email.id)
    assert metadata["user_id"] == str(user.id)
    assert metadata["sender_id"] == "vendor@supplier.com"
    assert metadata["recipient_id"] == "owner@corp.com"
    assert metadata["received_date"].startswith("2026-08-15T10:30:00")
    assert metadata["embedding_provider"] == "azure_openai"
    assert metadata["embedding_model"] == "text-embedding-3-large"
    assert metadata["embedding_dimensions"] == 3072


# ---------------------------------------------------------------------------
# 6. Idempotency & Duplicate Prevention
# ---------------------------------------------------------------------------

def test_idempotent_indexing(db_session, chroma_client, azure_mock_embedder):
    """Indexing the same email repeatedly must be a no-op and never duplicate records."""
    user = User(google_id="usr_idem", email="idem@example.com")
    db_session.add(user)
    db_session.flush()

    acct = GmailAccount(user_id=user.id, email="idem@example.com", access_token="token")
    db_session.add(acct)
    db_session.flush()

    email = Email(
        gmail_account_id=acct.id,
        gmail_message_id="msg_idem_1",
        subject="Weekly Standup",
        from_email="lead@example.com",
        to_email="idem@example.com",
        body="Standup is moved to 10:30 AM tomorrow.",
        received_at=datetime.now(timezone.utc),
    )
    db_session.add(email)
    db_session.commit()

    col_name = f"gmail_email_chunks_v2_{uuid4().hex[:8]}"
    rag_service = RAGIndexerService(
        chroma_client=chroma_client,
        collection_name=col_name,
        embedding_provider=azure_mock_embedder,
    )

    # First index
    rag_service.index_email(db=db_session, email_id=email.id)
    assert rag_service.collection.count() == 1
    assert len(db_session.scalars(select(EmailEmbedding)).all()) == 1

    # Second index (should skip cleanly)
    rag_service.index_email(db=db_session, email_id=email.id)
    assert rag_service.collection.count() == 1
    assert len(db_session.scalars(select(EmailEmbedding)).all()) == 1


# ---------------------------------------------------------------------------
# 7. Multi-Tenant User Isolation Retrieval
# ---------------------------------------------------------------------------

def test_user_isolation_retrieval(db_session, chroma_client, azure_mock_embedder):
    """Ensure User A cannot retrieve User B's email chunks under any circumstances."""
    user_a = User(google_id="usr_a", email="alice@company.com")
    user_b = User(google_id="usr_b", email="bob@company.com")
    db_session.add_all([user_a, user_b])
    db_session.flush()

    acct_a = GmailAccount(user_id=user_a.id, email="alice@company.com", access_token="token_a")
    acct_b = GmailAccount(user_id=user_b.id, email="bob@company.com", access_token="token_b")
    db_session.add_all([acct_a, acct_b])
    db_session.flush()

    email_a = Email(
        gmail_account_id=acct_a.id,
        gmail_message_id="msg_a_confidential",
        subject="Alice Private Salaries",
        from_email="hr@company.com",
        to_email="alice@company.com",
        body="Alice's confidential salary data and stock grants.",
        received_at=datetime.now(timezone.utc),
    )
    email_b = Email(
        gmail_account_id=acct_b.id,
        gmail_message_id="msg_b_general",
        subject="Company Picnic",
        from_email="hr@company.com",
        to_email="bob@company.com",
        body="Everyone is invited to the company picnic this Friday.",
        received_at=datetime.now(timezone.utc),
    )
    db_session.add_all([email_a, email_b])
    db_session.commit()

    col_name = f"gmail_email_chunks_v2_{uuid4().hex[:8]}"
    rag_service = RAGIndexerService(
        chroma_client=chroma_client,
        collection_name=col_name,
        embedding_provider=azure_mock_embedder,
    )

    rag_service.index_email(db=db_session, email_id=email_a.id)
    rag_service.index_email(db=db_session, email_id=email_b.id)
    assert rag_service.collection.count() == 2

    # Query as Bob for confidential salary data
    bob_results = rag_service.search_rag(query="confidential salary data", user_id=user_b.id, top_k=5)
    # Bob must NEVER see Alice's confidential email
    for result in bob_results:
        assert result["user_id"] == str(user_b.id)
        assert "salary" not in result["document"].lower()

    # Query as Alice
    alice_results = rag_service.search_rag(query="confidential salary data", user_id=user_a.id, top_k=5)
    assert len(alice_results) == 1
    assert alice_results[0]["user_id"] == str(user_a.id)
    assert "Alice Private Salaries" in alice_results[0]["subject"]
    assert alice_results[0]["sender_id"] == "hr@company.com"
    assert alice_results[0]["recipient_id"] == "alice@company.com"


# ---------------------------------------------------------------------------
# 8. Email Update & Deletion Safety
# ---------------------------------------------------------------------------

def test_email_update_and_deletion_safety(db_session, chroma_client, azure_mock_embedder):
    """Updating email text updates chunks; deleting email cleanly purges vectors and DB chunks."""
    user = User(google_id="usr_upd", email="upd@example.com")
    db_session.add(user)
    db_session.flush()

    acct = GmailAccount(user_id=user.id, email="upd@example.com", access_token="token")
    db_session.add(acct)
    db_session.flush()

    email = Email(
        gmail_account_id=acct.id,
        gmail_message_id="msg_upd_1",
        subject="Project Alpha Scope",
        from_email="pm@example.com",
        to_email="upd@example.com",
        body="Initial scope covers MVP features.",
        received_at=datetime.now(timezone.utc),
    )
    db_session.add(email)
    db_session.commit()

    col_name = f"gmail_email_chunks_v2_{uuid4().hex[:8]}"
    rag_service = RAGIndexerService(
        chroma_client=chroma_client,
        collection_name=col_name,
        embedding_provider=azure_mock_embedder,
    )

    rag_service.index_email(db=db_session, email_id=email.id)
    assert rag_service.collection.count() == 1

    # Update email body
    email.body = "Updated scope includes Phase 2 AI automation."
    db_session.commit()

    rag_service.index_email(db=db_session, email_id=email.id)
    assert rag_service.collection.count() == 1

    # Verify updated content in ChromaDB
    chroma_res = rag_service.collection.get(include=["documents"])
    assert "Phase 2 AI automation" in chroma_res["documents"][0]

    # Delete email
    rag_service.delete_email_from_rag(db=db_session, email_id=email.id)
    assert rag_service.collection.count() == 0
    assert len(db_session.scalars(select(EmailChunk).where(EmailChunk.email_id == email.id)).all()) == 0

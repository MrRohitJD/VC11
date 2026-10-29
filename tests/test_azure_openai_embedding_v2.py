"""Azure OpenAI configuration aliases and 3072-dimensional embedding verification."""

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

    for key in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"):
        monkeypatch.delenv(key, raising=False)
    test_settings = Settings(_env_file=None)
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

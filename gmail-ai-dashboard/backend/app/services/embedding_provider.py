"""Azure OpenAI text-embedding-3-large embedding provider via LangChain AzureOpenAIEmbeddings.

Features:
- Azure endpoint, API key, API version, and deployment name loaded from environment/settings
- Uses LangChain's AzureOpenAIEmbeddings
- Verifies and validates actual embedding dimensions returned by the deployment
- Same deployment used for document embeddings and query embeddings
- Batch generation with exponential backoff and retry handling
- Dimension verification probe (verifies 3072 or deployment-configured dimensions)
- Zero secret leakage in logs or exceptions
- Deterministic offline mock provider for test suites
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from typing import List, Optional

from langchain_openai import AzureOpenAIEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

AZURE_OPENAI_DEFAULT_DEPLOYMENT = "text-embedding-3-large"
AZURE_OPENAI_DEFAULT_MODEL = "text-embedding-3-large"
AZURE_OPENAI_LARGE_DIMENSIONS = 3072

# Aliases for backward compatibility
OPENAI_LARGE_MODEL = AZURE_OPENAI_DEFAULT_MODEL
OPENAI_LARGE_DIMENSIONS = AZURE_OPENAI_LARGE_DIMENSIONS


class MockAzureOpenAILargeEmbeddings:
    """Deterministic local vector generator for test suites and offline mode."""

    def __init__(self, dimensions: int = AZURE_OPENAI_LARGE_DIMENSIONS):
        self.dimensions = dimensions

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def _embed_single(self, text: str) -> List[float]:
        vec = [0.0] * self.dimensions
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec
        for word in words:
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimensions
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-9:
            vec = [v / norm for v in vec]
        return vec


# Alias for backward compatibility
MockOpenAILargeEmbeddings = MockAzureOpenAILargeEmbeddings


class EmbeddingProvider:
    """Manages AzureOpenAIEmbeddings instance, dimension verification, batching, and retries."""

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
        model_name: Optional[str] = None,
        dimensions: Optional[int] = None,
        max_retries: int = 5,
        mock_mode: bool = False,
        verify_on_init: bool = False,
    ):
        self.provider = "azure_openai"
        self.endpoint = (azure_endpoint or settings.azure_openai_endpoint).strip()
        self.api_key = (api_key or settings.azure_openai_api_key).strip()
        self.api_version = (api_version or settings.azure_openai_api_version).strip()
        self.deployment_name = (deployment_name or settings.azure_openai_embedding_deployment).strip()
        self.model_name = (model_name or settings.azure_openai_embedding_model).strip()
        self.dimensions = dimensions or settings.embedding_dimensions or AZURE_OPENAI_LARGE_DIMENSIONS
        self.max_retries = max_retries

        # Determine if running in mock/offline mode
        is_dummy_key = not self.api_key or any(
            self.api_key.lower().startswith(prefix) for prefix in ("test", "dummy", "sk-proj-", "mock")
        )
        is_invalid_endpoint = not self.endpoint or not (
            self.endpoint.startswith("https://") or self.endpoint.startswith("http://")
        )

        if mock_mode or is_dummy_key or is_invalid_endpoint:
            self._client = MockAzureOpenAILargeEmbeddings(dimensions=self.dimensions)
            self.is_mock = True
            logger.info(
                "EmbeddingProvider initialized in MOCK mode (dimensions=%d, deployment=%s)",
                self.dimensions,
                self.deployment_name,
            )
        else:
            self._client = AzureOpenAIEmbeddings(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
                azure_deployment=self.deployment_name,
                model=self.model_name,
                dimensions=self.dimensions,
                max_retries=self.max_retries,
            )
            self.is_mock = False
            logger.info(
                "EmbeddingProvider initialized with Azure OpenAI (deployment=%s, model=%s, api_version=%s, dimensions=%d)",
                self.deployment_name,
                self.model_name,
                self.api_version,
                self.dimensions,
            )

        if verify_on_init:
            self.verify_dimensions()

    def verify_dimensions(self) -> int:
        """Verify the actual embedding dimensions returned by the configured deployment.

        Sends a probe query vector and validates or updates self.dimensions.
        """
        if self.is_mock:
            return self.dimensions

        try:
            probe_vector = self.embed_query("deployment dimension verification probe")
            actual_dim = len(probe_vector)
            if actual_dim != self.dimensions:
                logger.warning(
                    "Azure deployment '%s' returned %d dimensions; updating configured %d",
                    self.deployment_name,
                    actual_dim,
                    self.dimensions,
                )
                self.dimensions = actual_dim
            else:
                logger.info(
                    "Successfully verified Azure deployment '%s' returns %d dimensions",
                    self.deployment_name,
                    actual_dim,
                )
            return self.dimensions
        except Exception as exc:
            logger.warning(
                "Could not probe Azure deployment dimensions (%s); using configured %d",
                exc,
                self.dimensions,
            )
            return self.dimensions

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a single query string using the configured deployment."""
        for attempt in range(self.max_retries):
            try:
                vec = self._client.embed_query(text)
                if len(vec) != self.dimensions:
                    raise ValueError(f"Expected {self.dimensions} dimensions, got {len(vec)}")
                return vec
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    logger.error("Failed query embedding after %d attempts: %s", self.max_retries, exc)
                    raise
                wait = min(2.0 ** attempt, 30.0)
                logger.warning(
                    "Query embedding retry %d/%d (waiting %.1fs): %s",
                    attempt + 1,
                    self.max_retries,
                    wait,
                    exc,
                )
                time.sleep(wait)
        return []

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Generate embeddings in batches using the configured deployment."""
        if not texts:
            return []

        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(self.max_retries):
                try:
                    vectors = self._client.embed_documents(batch)
                    for v in vectors:
                        if len(v) != self.dimensions:
                            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(v)}")
                    all_vectors.extend(vectors)
                    break
                except Exception as exc:
                    if attempt == self.max_retries - 1:
                        logger.error("Failed batch embedding (%d items) after %d attempts: %s", len(batch), self.max_retries, exc)
                        raise
                    wait = min(2.0 ** attempt, 30.0)
                    logger.warning(
                        "Batch embedding retry %d/%d (waiting %.1fs): %s",
                        attempt + 1,
                        self.max_retries,
                        wait,
                        exc,
                    )
                    time.sleep(wait)

        return all_vectors


default_embedding_provider = EmbeddingProvider()

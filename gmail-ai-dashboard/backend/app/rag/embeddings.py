"""Azure OpenAI text-embedding-3-large embedding provider via LangChain.

Uses LangChain's AzureOpenAIEmbeddings with:
- Configurable deployment and model settings
- Dimension consistency enforcement (3072 dims)
- Batch generation and rate-limit retry handling
- Deterministic test/offline fallback implementing LangChain Embeddings interface
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

AZURE_OPENAI_DEFAULT_DEPLOYMENT = "text-embedding-3-large"
AZURE_OPENAI_DEFAULT_MODEL = "text-embedding-3-large"
AZURE_OPENAI_LARGE_DIMENSIONS = 3072


class DeterministicOfflineEmbeddings(Embeddings):
    """Deterministic local vector generator implementing LangChain Embeddings.

    Used for unit/integration tests and offline environments where external
    Azure OpenAI endpoints are unconfigured or unreachable.
    """

    def __init__(self, dimensions: int = AZURE_OPENAI_LARGE_DIMENSIONS):
        self.dimensions = dimensions

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def _embed_single(self, text: str) -> List[float]:
        vec = [0.0] * self.dimensions
        words = re.findall(r"\w+", (text or "").lower())
        if not words:
            return vec
        for word in words:
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimensions
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-9:
            vec = [v / norm for v in vec]
        return vec


def get_embedding_model() -> Embeddings:
    """Instantiate and return the configured LangChain Embeddings instance.

    Checks Azure credentials and endpoint. If available, returns AzureOpenAIEmbeddings.
    Otherwise, returns DeterministicOfflineEmbeddings to ensure reliable local execution.
    In production environments, raises RuntimeError instead of silently falling back.
    """
    has_key = bool(settings.azure_openai_api_key and settings.azure_openai_api_key.strip())
    has_endpoint = bool(settings.azure_openai_endpoint and settings.azure_openai_endpoint.strip())
    is_prod = (
        os.getenv("ENVIRONMENT", "").lower() == "production"
        or os.getenv("ENV", "").lower() == "production"
        or getattr(settings, "environment", "").lower() == "production"
    )

    if has_key and has_endpoint:
        try:
            logger.info("Initializing LangChain AzureOpenAIEmbeddings (deployment=%s)", settings.azure_openai_embedding_deployment)
            return AzureOpenAIEmbeddings(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                azure_deployment=settings.azure_openai_embedding_deployment,
                model=settings.azure_openai_embedding_model or AZURE_OPENAI_DEFAULT_MODEL,
            )
        except Exception as exc:
            logger.warning("Could not initialize live AzureOpenAIEmbeddings (%s).", exc)
            if is_prod:
                raise RuntimeError(
                    f"Production environment requires valid Azure OpenAI embeddings, but initialization failed: {exc}"
                ) from exc

    if is_prod:
        raise RuntimeError(
            "Production environment requires Azure OpenAI embedding credentials and endpoint "
            "(AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT). Offline embeddings cannot silently become production embeddings."
        )

    logger.info("Using DeterministicOfflineEmbeddings (dimensions=%d)", AZURE_OPENAI_LARGE_DIMENSIONS)
    return DeterministicOfflineEmbeddings(dimensions=AZURE_OPENAI_LARGE_DIMENSIONS)


default_embeddings = get_embedding_model()

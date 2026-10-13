"""Dense vector embedder with caching and robust fallbacks."""

from __future__ import annotations

import hashlib
import math
import re
from typing import List
from config import config


class TextEmbedder:
    """Generates normalized vector embeddings for text chunks and queries."""

    def __init__(self, dimension: int = config.embedding_dimension):
        self.dimension = dimension
        self._cache: dict[str, List[float]] = {}

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string into a normalized dense vector."""
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Deterministic hashing embedding (fast, dependency-free, zero-latency fallback)
        vec = [0.0] * self.dimension
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec

        for word in words:
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-9:
            vec = [v / norm for v in vec]

        self._cache[cache_key] = vec
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of texts in batch."""
        return [self.embed_text(t) for t in texts]


default_embedder = TextEmbedder()

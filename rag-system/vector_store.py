"""Vector store for indexing, storing, and searching email chunks."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
from schemas import EmailChunk, EmailMetadata


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two normalized vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(0.0, min(1.0, (dot + 1.0) / 2.0))  # Scale to [0, 1]


class VectorStore:
    """In-memory vector store with persistence and metadata filtering."""

    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = persistence_path
        self.chunks: Dict[str, EmailChunk] = {}

    def add_chunks(self, chunks: List[EmailChunk]) -> int:
        """Add chunks to the vector store."""
        added = 0
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            added += 1
        return added

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_label: Optional[str] = None,
        filter_sender: Optional[str] = None,
    ) -> List[Tuple[EmailChunk, float]]:
        """Search for top_k most similar chunks matching metadata filters."""
        results: List[Tuple[EmailChunk, float]] = []

        for chunk in self.chunks.values():
            if not chunk.embedding:
                continue

            # Apply metadata filters
            if filter_label and chunk.metadata.ai_label != filter_label:
                continue
            if filter_sender and filter_sender.lower() not in (chunk.metadata.sender_email or "").lower():
                continue

            score = cosine_similarity(query_embedding, chunk.embedding)
            results.append((chunk, score))

        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        self.chunks.clear()

    def count(self) -> int:
        return len(self.chunks)


default_vector_store = VectorStore()

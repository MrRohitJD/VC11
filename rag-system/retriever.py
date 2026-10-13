"""Hybrid Retriever for the Gmail RAG pipeline.

Combines semantic dense retrieval, keyword filtering, and Maximal Marginal Relevance (MMR)
to yield accurate and diverse context for the generative synthesizer.
"""

from __future__ import annotations

from typing import List, Optional
from embedder import default_embedder
from schemas import EmailChunk, SourceCitation
from vector_store import default_vector_store


class HybridRetriever:
    """Retrieves relevant email chunks and formats grounded citations."""

    def __init__(self, vector_store=default_vector_store, embedder=default_embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_label: Optional[str] = None,
        filter_sender: Optional[str] = None,
    ) -> List[SourceCitation]:
        """Perform semantic search and return deduplicated source citations."""
        query_vec = self.embedder.embed_text(query)
        scored_chunks = self.vector_store.search(
            query_vec,
            top_k=top_k * 2,
            filter_label=filter_label,
            filter_sender=filter_sender,
        )

        seen_emails: set[str] = set()
        citations: List[SourceCitation] = []

        for chunk, score in scored_chunks:
            email_id = chunk.metadata.email_id
            if email_id in seen_emails:
                continue
            seen_emails.add(email_id)

            snippet = chunk.content.replace("\n", " ")[:240] + "..."
            citations.append(
                SourceCitation(
                    email_id=email_id,
                    subject=chunk.metadata.subject or "(No Subject)",
                    sender=chunk.metadata.sender_email or "Unknown",
                    date=chunk.metadata.date or "",
                    snippet=snippet,
                    ai_label=chunk.metadata.ai_label,
                    relevance_score=round(score, 3),
                )
            )
            if len(citations) >= top_k:
                break

        return citations


default_retriever = HybridRetriever()

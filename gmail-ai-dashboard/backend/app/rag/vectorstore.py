"""ChromaDB vector store integration using LangChain Chroma.

Features:
- Persistent disk storage at ./data/chroma
- Collection 'gmail_email_chunks' with cosine similarity
- Stable chunk ID generation: f"{email_id}_{chunk_index}"
- Multi-tenant user isolation enforced on every query
- LangChain VectorStoreRetriever interface
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.config import settings
from app.rag.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "gmail_email_chunks"


class ChromaVectorStoreManager:
    """Manages persistent ChromaDB vector store through LangChain."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
        embedding_model: Optional[Embeddings] = None,
    ):
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name_v1 or DEFAULT_COLLECTION_NAME
        self.embedding_function = embedding_function or embedding_model or get_embedding_model()

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self._vector_store: Optional[Chroma] = None

    @property
    def vector_store(self) -> Chroma:
        """Lazy-initialize and return the LangChain Chroma instance."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
            )
        return self._vector_store

    def add_chunks(self, chunks: List[Document]) -> List[str]:
        """Add LangChain document chunks to ChromaDB with deterministic IDs."""
        if not chunks:
            return []

        ids: List[str] = []
        for chunk in chunks:
            email_id = chunk.metadata.get("email_id", "email")
            chunk_idx = chunk.metadata.get("chunk_index", 0)
            chunk_id = f"{email_id}_{chunk_idx}"
            ids.append(chunk_id)

        # Chroma stores string/int/float/bool primitives in metadata
        sanitized_chunks: List[Document] = []
        for chunk in chunks:
            sanitized_meta = {}
            for k, v in chunk.metadata.items():
                if v is None:
                    sanitized_meta[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    sanitized_meta[k] = v
                else:
                    sanitized_meta[k] = str(v)
            sanitized_chunks.append(Document(page_content=chunk.page_content, metadata=sanitized_meta))

        self.vector_store.add_documents(documents=sanitized_chunks, ids=ids)
        return ids

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """Convenience method to add LangChain documents to ChromaDB."""
        if ids:
            self.vector_store.add_documents(documents=documents, ids=ids)
            return ids
        return self.add_chunks(documents)

    def search(
        self,
        query: str,
        user_id: UUID | str,
        top_k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> List[Document]:
        """Search similar chunks with mandatory user isolation."""
        user_filter = {"user_id": str(user_id)}
        combined_filter = user_filter
        if filter_dict:
            combined_filter = {"$and": [user_filter, filter_dict]}

        return self.vector_store.similarity_search(
            query=query,
            k=top_k,
            filter=combined_filter,
        )

    def search_with_scores(
        self,
        query: str,
        user_id: UUID | str,
        top_k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Search similar chunks returning relevance scores."""
        user_filter = {"user_id": str(user_id)}
        combined_filter = user_filter
        if filter_dict:
            combined_filter = {"$and": [user_filter, filter_dict]}

        return self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=top_k,
            filter=combined_filter,
        )

    def delete_by_email_id(self, email_id: UUID | str) -> None:
        """Delete all chunks associated with an email from ChromaDB."""
        try:
            self.vector_store.delete(where={"email_id": str(email_id)})
        except Exception as exc:
            logger.debug("Chroma deletion for email %s: %s", email_id, exc)

    def count(self) -> int:
        """Return the total number of items stored in ChromaDB."""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0


default_chroma_manager = ChromaVectorStoreManager()
default_vectorstore = default_chroma_manager

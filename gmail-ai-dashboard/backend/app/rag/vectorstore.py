"""Vector-store compatibility facade.

All vector data is stored in PostgreSQL's ``email_embeddings.embedding``
column. ``ChromaVectorStoreManager`` remains as a source-compatible name for
older callers, but it is a PostgreSQL manager and never creates or opens a
local Chroma database.
"""

from __future__ import annotations

from app.rag.pgvector_store import PostgresVectorStoreManager


class ChromaVectorStoreManager(PostgresVectorStoreManager):
    """Deprecated compatibility name for the PostgreSQL vector manager."""

    def __init__(self, *args, persist_directory=None, collection_name=None, **kwargs):
        # The legacy arguments are intentionally accepted and ignored. This
        # prevents old integrations from silently reintroducing local storage.
        super().__init__(**kwargs)


default_vectorstore = PostgresVectorStoreManager()
# Keep this name temporarily because retrievers and integrations imported it.
# It points to PostgreSQL, never to ChromaDB.
default_chroma_manager = default_vectorstore

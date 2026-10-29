"""Default PostgreSQL vector store."""

from app.rag.pgvector_store import PostgresVectorStoreManager

default_vectorstore = PostgresVectorStoreManager()

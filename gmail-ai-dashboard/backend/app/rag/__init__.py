"""LangChain Gmail RAG and AI Assistant module."""

from app.rag.agent import GmailAIAssistant, default_assistant
from app.rag.chains import GroundedRAGChain
from app.rag.chunking import EmailChunker
from app.rag.embeddings import get_embedding_model
from app.rag.indexing import RAGIndexingPipeline, default_pipeline
from app.rag.loaders import GmailDocumentAdapter
from app.rag.memory import PostgresChatMessageHistory, get_or_create_chat_session
from app.rag.retrieval import (
    ChromaSemanticRetriever,
    GmailCustomHybridRetriever,
    GmailHybridRetriever,
    HistoryAwareRetriever,
    MultiQueryRetriever,
    RetrievedCitation,
    default_retriever,
    documents_to_citations,
)
from app.rag.tools import create_gmail_tools
from app.rag.vectorstore import ChromaVectorStoreManager, default_vectorstore

__all__ = [
    "GmailAIAssistant",
    "default_assistant",
    "GroundedRAGChain",
    "EmailChunker",
    "get_embedding_model",
    "RAGIndexingPipeline",
    "default_pipeline",
    "GmailDocumentAdapter",
    "PostgresChatMessageHistory",
    "get_or_create_chat_session",
    "ChromaSemanticRetriever",
    "GmailCustomHybridRetriever",
    "GmailHybridRetriever",
    "HistoryAwareRetriever",
    "MultiQueryRetriever",
    "documents_to_citations",
    "RetrievedCitation",
    "default_retriever",
    "create_gmail_tools",
    "ChromaVectorStoreManager",
    "default_vectorstore",
]

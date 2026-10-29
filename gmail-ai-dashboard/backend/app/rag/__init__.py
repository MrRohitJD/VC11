"""Lazy exports: mailbox helpers must not initialize the entire AI stack."""
from importlib import import_module

_EXPORTS = {
    'GmailAIAssistant': 'agent', 'default_assistant': 'agent',
    'GroundedRAGChain': 'chains', 'EmailChunker': 'chunking',
    'get_embedding_model': 'embeddings',
    'RAGIndexingPipeline': 'indexing', 'default_pipeline': 'indexing',
    'GmailDocumentAdapter': 'loaders',
    'PostgresChatMessageHistory': 'memory', 'get_or_create_chat_session': 'memory',
    'PostgresSemanticRetriever': 'retrieval', 'GmailCustomHybridRetriever': 'retrieval',
    'GmailHybridRetriever': 'retrieval', 'HistoryAwareRetriever': 'retrieval',
    'MultiQueryRetriever': 'retrieval', 'RetrievedCitation': 'retrieval',
    'default_retriever': 'retrieval', 'documents_to_citations': 'retrieval',
    'create_gmail_tools': 'tools',
    'PostgresVectorStoreManager': 'pgvector_store', 'default_vectorstore': 'vectorstore',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(f'.{_EXPORTS[name]}', __name__), name)
    globals()[name] = value
    return value

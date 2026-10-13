"""Configuration settings for the RAG System."""

import os
from dataclasses import dataclass


@dataclass
class RAGConfig:
    # Embedding Configuration
    embedding_dimension: int = 384
    embedding_model_name: str = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Chunking Configuration
    max_chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval Configuration
    top_k_default: int = 5
    similarity_threshold: float = 0.35
    use_mmr: bool = True
    mmr_diversity_bias: float = 0.3

    # Storage Configuration
    vector_store_path: str = os.getenv("RAG_STORAGE_PATH", "./data/vector_store.json")

    # LLM Settings for Synthesis
    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "nvidia")
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    azure_openai_key: str = os.getenv("AZURE_OPENAI_KEY", "")


config = RAGConfig()

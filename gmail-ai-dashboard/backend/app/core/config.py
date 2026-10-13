from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/


class Settings(BaseSettings):
    database_url: str = "sqlite:///./gmail_ai_dashboard.db"
    database_pool_size: int = Field(5, validation_alias=AliasChoices("DATABASE_POOL_SIZE"))
    database_max_overflow: int = Field(10, validation_alias=AliasChoices("DATABASE_MAX_OVERFLOW"))
    google_client_id: str = Field("", validation_alias=AliasChoices("GOOGLE_CLIENT_ID", "OAUTH_GOOGLE_CLIENT_ID"))
    google_client_secret: str = Field("", validation_alias=AliasChoices("GOOGLE_CLIENT_SECRET", "OAUTH_GOOGLE_CLIENT_SECRET"))
    oauth_redirect_uri: str = Field("http://localhost:8000/auth/google/callback", validation_alias=AliasChoices("OAUTH_GOOGLE_REDIRECT_URI", "GOOGLE_REDIRECT_URI"))
    backend_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:5173"
    session_secret: str = "change-this-session-secret"
    secure_cookies: bool = False

    # LLM & Embedding provider keys
    nvidia_api_key: str = Field("", validation_alias=AliasChoices("NVIDIA_API_KEY"))
    azure_openai_api_key: str = Field("", validation_alias=AliasChoices("AZURE_OPENAI_API_KEY", "AZURE_API_KEY"))
    azure_openai_endpoint: str = Field("", validation_alias=AliasChoices("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_BASE", "AZURE_ENDPOINT"))
    azure_openai_api_version: str = Field("2024-02-01", validation_alias=AliasChoices("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION"))
    azure_openai_embedding_deployment: str = Field(
        "text-embedding-3-large",
        validation_alias=AliasChoices(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_EMBEDDING_DEPLOYMENT",
            "AZURE_OPENAI_DEPLOYMENT",
        ),
    )
    azure_openai_embedding_model: str = Field(
        "text-embedding-3-large",
        validation_alias=AliasChoices("AZURE_OPENAI_EMBEDDING_MODEL", "AZURE_EMBEDDING_MODEL"),
    )
    openai_api_key: str = Field("", validation_alias=AliasChoices("OPENAI_API_KEY"))
    explabs_api_key: str = Field("", validation_alias=AliasChoices("EXPLABS_API_KEY"))

    # ChromaDB & RAG configuration
    chroma_persist_dir: str = Field(str(_BACKEND_DIR / "data" / "chroma"), validation_alias=AliasChoices("CHROMA_PERSIST_DIR", "RAG_STORAGE_PATH"))
    chroma_collection_name_v1: str = "gmail_email_chunks"
    chroma_collection_name_v2: str = Field("gmail_email_chunks_v2", validation_alias=AliasChoices("CHROMA_COLLECTION_NAME_V2", "CHROMA_COLLECTION_NAME"))
    active_chroma_collection: str = Field("gmail_email_chunks_v2", validation_alias=AliasChoices("ACTIVE_CHROMA_COLLECTION"))
    embedding_provider: str = Field("azure_openai", validation_alias=AliasChoices("RAG_EMBEDDING_PROVIDER", "EMBEDDING_PROVIDER"))
    embedding_model_name: str = Field("text-embedding-3-large", validation_alias=AliasChoices("RAG_EMBEDDING_MODEL", "EMBEDDING_MODEL"))
    embedding_dimensions: int = Field(3072, validation_alias=AliasChoices("RAG_EMBEDDING_DIMENSIONS", "EMBEDDING_DIMENSIONS"))

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        extra="ignore",
    )


settings = Settings()

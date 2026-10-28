from pathlib import Path

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/


class Settings(BaseSettings):
    gmail_mcp_enabled: bool = True
    gmail_mcp_max_messages: int = Field(1000, ge=1, le=10000)
    # Keep live discovery bounded so a slow Gmail/API process cannot hold a
    # mailbox answer open for minutes.  The client also applies a hard cap.
    gmail_mcp_timeout_seconds: int = Field(30, ge=10, le=600)
    langgraph_checkpoint_path: str = str(_BACKEND_DIR / "data" / "langgraph.sqlite")
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "mailbox-devLG"
    database_url: str = "sqlite:///./gmail_ai_dashboard.db"
    # PostgreSQL pgvector is the sole vector backend.
    vector_backend: str = Field("postgres", validation_alias=AliasChoices("RAG_VECTOR_BACKEND"))
    gmail_sync_limit: int = Field(500, ge=1, le=5000, validation_alias=AliasChoices("GMAIL_SYNC_LIMIT"))
    database_pool_size: int = Field(5, validation_alias=AliasChoices("DATABASE_POOL_SIZE"))
    database_max_overflow: int = Field(10, validation_alias=AliasChoices("DATABASE_MAX_OVERFLOW"))
    google_client_id: str = Field("", validation_alias=AliasChoices("GOOGLE_CLIENT_ID", "OAUTH_GOOGLE_CLIENT_ID"))
    google_client_secret: str = Field("", validation_alias=AliasChoices("GOOGLE_CLIENT_SECRET", "OAUTH_GOOGLE_CLIENT_SECRET"))
    oauth_redirect_uri: str = Field("http://localhost:8000/api/auth/callback", validation_alias=AliasChoices("OAUTH_GOOGLE_REDIRECT_URI", "GOOGLE_REDIRECT_URI"))
    backend_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:5173"
    session_secret: str = "change-this-session-secret"
    secure_cookies: bool = False

    # LLM & Embedding provider keys
    nvidia_api_key: str = Field("", validation_alias=AliasChoices("NVIDIA_API_KEY"))
    azure_openai_api_key: str = Field("", validation_alias=AliasChoices("AZURE_OPENAI_API_KEY", "AZURE_API_KEY"))
    azure_openai_endpoint: str = Field("", validation_alias=AliasChoices("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_BASE", "AZURE_ENDPOINT"))
    azure_openai_api_version: str = Field("2024-02-01", validation_alias=AliasChoices("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION"))
    azure_openai_deployment: str = Field("gpt-4.1-mini", validation_alias=AliasChoices("AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_CHAT_DEPLOYMENT"))
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
    llm_request_timeout_seconds: float = Field(
        20.0,
        validation_alias=AliasChoices("LLM_REQUEST_TIMEOUT_SECONDS", "LLM_TIMEOUT"),
    )
    llm_max_retries: int = Field(
        0,
        validation_alias=AliasChoices("LLM_MAX_RETRIES"),
    )
    # Keep interactive mailbox answers bounded.  These defaults are deliberately
    # conservative: a grounded answer should be concise, while long provider
    # retries are more damaging to UX than a short, source-backed response.
    llm_max_completion_tokens: int = Field(
        1600,
        ge=256,
        le=8192,
        validation_alias=AliasChoices("LLM_MAX_COMPLETION_TOKENS"),
    )
    rag_relevance_parallelism: int = Field(
        2,
        ge=1,
        le=8,
        validation_alias=AliasChoices("RAG_RELEVANCE_PARALLELISM"),
    )
    rag_vector_candidate_multiplier: int = Field(
        4, ge=1, le=10, validation_alias=AliasChoices("RAG_VECTOR_CANDIDATE_MULTIPLIER"),
    )

    embedding_provider: str = Field("azure_openai", validation_alias=AliasChoices("RAG_EMBEDDING_PROVIDER", "EMBEDDING_PROVIDER"))
    embedding_model_name: str = Field("text-embedding-3-large", validation_alias=AliasChoices("RAG_EMBEDDING_MODEL", "EMBEDDING_MODEL"))
    embedding_dimensions: int = Field(3072, validation_alias=AliasChoices("RAG_EMBEDDING_DIMENSIONS", "EMBEDDING_DIMENSIONS"))

    @field_validator("vector_backend")
    @classmethod
    def postgres_only_vector_backend(cls, value: str) -> str:
        normalized = (value or "postgres").strip().lower()
        # Older deployments may still carry RAG_VECTOR_BACKEND=chroma in
        # their environment. Chroma is no longer used, but that stale value
        # should not prevent the API from starting after the pgvector
        # migration.
        if normalized in {"", "chroma", "chromadb", "pgvector"}:
            return "postgres"
        if normalized != "postgres":
            raise ValueError("Only PostgreSQL pgvector is supported; local ChromaDB storage has been removed")
        return normalized

    @field_validator("database_url")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql+psycopg://" + value[len("postgres://"):]
        if value.startswith("postgresql://"):
            return "postgresql+psycopg://" + value[len("postgresql://"):]
        return value

    @model_validator(mode="after")
    def align_local_vector_backend(self):
        """Use the local vector store when the configured database is SQLite."""
        if self.database_url.startswith("sqlite") and self.vector_backend == "postgres":
            self.vector_backend = "chroma"
        return self

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        extra="ignore",
    )


settings = Settings()

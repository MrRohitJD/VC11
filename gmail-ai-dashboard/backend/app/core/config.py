from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/


class Settings(BaseSettings):
	database_url: str = "sqlite:///./gmail_ai_dashboard.db"
	google_client_id: str = Field("", validation_alias=AliasChoices("GOOGLE_CLIENT_ID", "OAUTH_GOOGLE_CLIENT_ID"))
	google_client_secret: str = Field("", validation_alias=AliasChoices("GOOGLE_CLIENT_SECRET", "OAUTH_GOOGLE_CLIENT_SECRET"))
	oauth_redirect_uri: str = Field("http://localhost:8000/auth/google/callback", validation_alias=AliasChoices("OAUTH_GOOGLE_REDIRECT_URI", "GOOGLE_REDIRECT_URI"))
	backend_url: str = "http://localhost:8000"
	frontend_url: str = "http://localhost:5173"
	session_secret: str = "change-this-session-secret"
	secure_cookies: bool = False

	model_config = SettingsConfigDict(
		env_file=str(_BACKEND_DIR / ".env"),
		extra="ignore",
	)


settings = Settings()

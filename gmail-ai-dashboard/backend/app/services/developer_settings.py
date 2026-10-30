"""Per-account developer credentials; plaintext never enters API responses."""

from __future__ import annotations

import base64
import hashlib
from contextvars import ContextVar

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import inspect
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.core.config import settings

database_url_context: ContextVar[str | None] = ContextVar("developer_database_url", default=None)
llm_key_context: ContextVar[tuple[str, str] | None] = ContextVar("developer_llm_key", default=None)


def _cipher() -> Fernet:
    secret = settings.developer_settings_encryption_key or settings.session_secret
    if secret == "change-this-session-secret" or len(secret) < 32:
        raise ValueError("Set a strong DEVELOPER_SETTINGS_ENCRYPTION_KEY or SESSION_SECRET before saving developer credentials")
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt(value: str) -> str:
    return _cipher().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return _cipher().decrypt(value.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Developer credential cannot be decrypted; check the server encryption secret") from exc


def normalized_database_url(value: str) -> str:
    value = value.strip()
    if value.startswith("postgres://"):
        value = "postgresql+psycopg://" + value[len("postgres://"):]
    elif value.startswith("postgresql://"):
        value = "postgresql+psycopg://" + value[len("postgresql://"):]
    url = make_url(value)
    if url.drivername != "postgresql+psycopg" or not url.host or not url.database:
        raise ValueError("Use a PostgreSQL connection URL with a host and database name")
    return url.render_as_string(hide_password=False)


def public_settings(user) -> dict:
    url = decrypt(user.developer_database_url)
    display_url = None
    if url:
        parsed = make_url(url)
        host = parsed.host or ""
        username = f"{parsed.username}:••••@" if parsed.username else ""
        display_url = f"postgresql://{username}{host}:{parsed.port or 5432}/{parsed.database}"
    return {
        "database_configured": bool(url),
        "database_url_display": display_url,
        "llm_provider": user.developer_llm_provider,
        "llm_key_configured": bool(user.developer_llm_api_key),
        "daily_chat_limit": settings.developer_daily_chat_limit if user.developer_llm_api_key else 50,
    }


def verify_personal_database(url: str) -> None:
    """Require an already migrated instance; never run DDL on a supplied URL."""
    from app.database.connection import _personal_engine
    from app.models.email import Base

    engine = _personal_engine(url)
    # Bind the inspector to one connection. With NullPool, inspecting each table
    # through the engine starts a new remote TLS session for every query.
    with engine.connect() as connection:
        inspector = inspect(connection)
        existing = set(inspector.get_table_names())
        if set(Base.metadata.tables) - existing:
            raise ValueError("Database connected, but the app schema is missing. Download and apply the database schema in Supabase SQL Editor, then try again")
        for table in Base.metadata.tables.values():
            expected = {column.name for column in table.columns}
            actual = {column["name"] for column in inspector.get_columns(table.name)}
            if expected - actual:
                raise ValueError("Database connected, but the app schema is out of date. Apply this project's Alembic migrations, then try again")


def ensure_personal_identity(central_user, url: str, *, refresh_authorization: bool = False) -> None:
    """Seed identity and Gmail authorization; never copy mailbox or RAG data."""
    from sqlalchemy.exc import IntegrityError
    from app.database.connection import _personal_engine
    from app.models.email import GmailAccount, User

    # Concurrent first requests can both observe an empty personal database.
    # Retry once after the other request commits its identity rows.
    for attempt in range(2):
        try:
            with Session(_personal_engine(url)) as target:
                user = target.get(User, central_user.id)
                if user is None:
                    user = User(id=central_user.id, google_id=central_user.google_id,
                                email=central_user.email, name=central_user.name,
                                picture=central_user.picture, is_demo=central_user.is_demo)
                    target.add(user)
                    target.flush()
                for source_account in central_user.gmail_accounts:
                    account = target.get(GmailAccount, source_account.id)
                    if account is None:
                        account = GmailAccount(
                            id=source_account.id, user_id=central_user.id,
                            email=source_account.email, provider=source_account.provider,
                            gmail_user_id=source_account.gmail_user_id,
                            access_token=source_account.access_token,
                            refresh_token=source_account.refresh_token,
                            token_expiry=source_account.token_expiry,
                            scope=source_account.scope,
                        )
                        target.add(account)
                    elif refresh_authorization:
                        for field in ("access_token", "refresh_token", "token_expiry", "scope"):
                            setattr(account, field, getattr(source_account, field))
                target.commit()
            return
        except IntegrityError:
            if attempt:
                raise

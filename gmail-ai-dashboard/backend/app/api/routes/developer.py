"""Authenticated developer settings for personal database and LLM credentials."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.connection import CentralSessionLocal
from app.models.email import User
from app.models.email import Base as EmailBase
from app.models.rag import Base as RagBase
import app.models.chat_quota  # noqa: F401
import app.models.llm  # noqa: F401
from app.services.developer_settings import (
    encrypt, normalized_database_url, public_settings,
    verify_personal_database,
)

router = APIRouter(prefix="/api/developer", tags=["developer"])


def _schema_sql() -> str:
    """Render the current ORM schema as a portable PostgreSQL bootstrap file."""
    from sqlalchemy.schema import CreateIndex, CreateTable
    from sqlalchemy.dialects import postgresql

    tables = list(EmailBase.metadata.sorted_tables)
    dialect = postgresql.dialect()
    lines = [
        "-- Gmail AI Dashboard developer database schema",
        "-- Generated from the current application models.",
        "CREATE EXTENSION IF NOT EXISTS vector;",
        "",
    ]
    for table in tables:
        lines.append(str(CreateTable(table).compile(dialect=dialect)) + ";")
    for table in tables:
        for index in table.indexes:
            lines.append(str(CreateIndex(index).compile(dialect=dialect)) + ";")
    lines.append("\n-- Keep this database current with: alembic upgrade head")
    return "\n\n".join(lines) + "\n"


def central_db():
    with CentralSessionLocal() as db:
        yield db


class DeveloperSettingsUpdate(BaseModel):
    database_url: str | None = Field(default=None, max_length=2048)
    llm_provider: Literal["google", "openai"] | None = None
    llm_api_key: str | None = Field(default=None, max_length=4096)
    clear_database: bool = False
    clear_llm_key: bool = False


@router.get("")
def read_developer_settings(user: User = Depends(get_current_user),
                            db: Session = Depends(central_db)):
    if user.is_demo:
        raise HTTPException(403, "Developer settings are unavailable in the demo")
    return public_settings(db.get(User, user.id))


@router.get("/schema.sql", include_in_schema=False)
def download_schema():
    return Response(
        content=_schema_sql(),
        media_type="application/sql",
        headers={"Content-Disposition": 'attachment; filename="gmail-ai-dashboard-schema.sql"'},
    )


@router.put("")
def update_developer_settings(body: DeveloperSettingsUpdate,
                              user: User = Depends(get_current_user),
                              db: Session = Depends(central_db)):
    if user.is_demo:
        raise HTTPException(403, "Developer settings are unavailable in the demo")
    owner = db.get(User, user.id)
    if ((body.database_url and body.database_url.strip()) or
            (body.llm_api_key and body.llm_api_key.strip())):
        try:
            encrypt("readiness-check")
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
    if body.clear_database:
        owner.developer_database_url = None
    elif body.database_url is not None and body.database_url.strip():
        try:
            url = normalized_database_url(body.database_url)
            verify_personal_database(url)
        except (ValueError, SQLAlchemyError, OSError) as exc:
            # Never return a driver exception: it may contain the password.
            if isinstance(exc, ValueError):
                raise HTTPException(400, str(exc)) from exc
            if isinstance(exc, OperationalError):
                message = str(exc.orig).lower()
                if any(term in message for term in ("timeout", "timed out", "network is unreachable", "could not translate host", "name or service not known")):
                    raise HTTPException(400, "Could not reach the database. For Supabase, try the Session pooler connection string from Connect (port 5432), especially if this server has no IPv6 access") from exc
                if any(term in message for term in ("password authentication failed", "invalid password", "invalid user", "tenant or user not found")):
                    raise HTTPException(400, "Database authentication failed. Check the username and password in the connection string") from exc
            raise HTTPException(400, "Could not connect to the database") from exc
        try:
            owner.developer_database_url = encrypt(url)
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc

    if body.clear_llm_key:
        owner.developer_llm_provider = None
        owner.developer_llm_api_key = None
    elif body.llm_api_key is not None and body.llm_api_key.strip():
        provider = body.llm_provider or owner.developer_llm_provider
        if provider not in {"google", "openai"}:
            raise HTTPException(400, "Choose Google Gemini or OpenAI before saving a key")
        owner.developer_llm_provider = provider
        try:
            owner.developer_llm_api_key = encrypt(body.llm_api_key.strip())
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
    elif body.llm_provider and body.llm_provider != owner.developer_llm_provider:
        raise HTTPException(400, "Enter a key when changing providers")

    db.commit()
    return public_settings(owner)

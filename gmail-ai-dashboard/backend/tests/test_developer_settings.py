"""Credential masking, per-request routing and personal provider isolation."""

from types import SimpleNamespace
from uuid import uuid4

from fastapi import Request
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database import connection
from app.rag import chains
from app.services import developer_settings as dev
from app.services.chat_quota import _identity_and_limits
from app.models.email import Email, GmailAccount, User
from app.api.routes import developer as developer_route
from app.api.routes.developer import DeveloperSettingsUpdate, update_developer_settings


def test_encrypted_credentials_are_never_returned(monkeypatch):
    monkeypatch.setattr(settings, "developer_settings_encryption_key", "test-secret-with-more-than-thirty-two-characters")
    url = "postgresql+psycopg://alice:private-password@db.example.com:5432/mail"
    owner = SimpleNamespace(
        developer_database_url=dev.encrypt(url),
        developer_llm_provider="openai",
        developer_llm_api_key=dev.encrypt("sk-private-value"),
    )
    assert "private-password" not in owner.developer_database_url
    assert dev.decrypt(owner.developer_database_url) == url
    shown = dev.public_settings(owner)
    assert shown["database_configured"] is True
    assert "private-password" not in str(shown)
    assert "alice:••••@db.example.com:5432/mail" in shown["database_url_display"]
    assert "sk-private-value" not in str(shown)
    assert shown["llm_key_configured"] is True
    assert shown["daily_chat_limit"] == 500


def test_only_postgresql_urls_are_accepted():
    assert dev.normalized_database_url("postgres://alice:secret@db.example.com/mail").startswith(
        "postgresql+psycopg://")
    with pytest.raises(ValueError):
        dev.normalized_database_url("sqlite:///private.db")


def test_personal_database_context_changes_session_binding(monkeypatch):
    personal_engine = create_engine("sqlite:///:memory:")
    monkeypatch.setattr(connection, "_personal_engine", lambda url: personal_engine)
    token = dev.database_url_context.set("postgresql+psycopg://example.com/personal")
    try:
        with connection.SessionLocal() as db:
            assert db.get_bind() is personal_engine
    finally:
        dev.database_url_context.reset(token)
        personal_engine.dispose()


def test_personal_database_receives_only_identity_and_gmail_authorization(monkeypatch):
    personal_engine = create_engine("sqlite:///:memory:")
    User.__table__.create(personal_engine)
    GmailAccount.__table__.create(personal_engine)
    Email.__table__.create(personal_engine)
    monkeypatch.setattr(connection, "_personal_engine", lambda url: personal_engine)
    source = User(id=uuid4(), google_id="google-1", email="owner@example.com", name="Owner")
    source.gmail_accounts.append(GmailAccount(
        id=uuid4(), user_id=source.id, email=source.email,
        access_token="oauth-token", refresh_token="refresh-token"))
    source.gmail_accounts[0].emails.append(Email(gmail_message_id="existing-app-mail"))
    dev.ensure_personal_identity(source, "postgresql+psycopg://example.com/personal")
    with Session(personal_engine) as db:
        copied = db.get(User, source.id)
        assert copied.email == source.email
        assert copied.gmail_accounts[0].refresh_token == "refresh-token"
        assert db.query(Email).count() == 0
    personal_engine.dispose()


def test_settings_endpoint_saves_only_an_encrypted_provider_key(monkeypatch):
    monkeypatch.setattr(settings, "developer_settings_encryption_key", "test-secret-with-more-than-thirty-two-characters")
    engine = create_engine("sqlite:///:memory:")
    User.__table__.create(engine)
    with Session(engine) as db:
        owner = User(id=uuid4(), google_id="google-2", email="key@example.com")
        db.add(owner)
        db.commit()
        response = update_developer_settings(
            DeveloperSettingsUpdate(llm_provider="google", llm_api_key="private-key"),
            owner, db)
        assert response["llm_provider"] == "google"
        assert "private-key" not in str(response)
        assert db.get(User, owner.id).developer_llm_api_key != "private-key"
        assert dev.decrypt(owner.developer_llm_api_key) == "private-key"
    engine.dispose()


def test_connecting_database_only_verifies_and_saves_url(monkeypatch):
    monkeypatch.setattr(settings, "developer_settings_encryption_key", "test-secret-with-more-than-thirty-two-characters")
    checked = []
    monkeypatch.setattr(developer_route, "verify_personal_database", checked.append)
    monkeypatch.setattr(connection, "_personal_engine", lambda url: (_ for _ in ()).throw(AssertionError("Personal database must not be written during connection")))
    engine = create_engine("sqlite:///:memory:")
    User.__table__.create(engine)
    with Session(engine) as db:
        owner = User(id=uuid4(), google_id="google-connect", email="connect@example.com")
        db.add(owner)
        db.commit()
        response = update_developer_settings(
            DeveloperSettingsUpdate(database_url="postgresql://alice:secret@db.example.com/mail"),
            owner, db)
        assert response["database_configured"] is True
        assert checked == ["postgresql+psycopg://alice:secret@db.example.com/mail"]
    engine.dispose()


def test_disconnecting_only_clears_the_selected_database(monkeypatch):
    monkeypatch.setattr(settings, "developer_settings_encryption_key", "test-secret-with-more-than-thirty-two-characters")
    engine = create_engine("sqlite:///:memory:")
    User.__table__.create(engine)
    with Session(engine) as db:
        owner = User(id=uuid4(), google_id="google-disconnect", email="disconnect@example.com")
        owner.developer_database_url = dev.encrypt("postgresql+psycopg://alice:secret@db.example.com/mail")
        db.add(owner)
        db.commit()
        response = update_developer_settings(DeveloperSettingsUpdate(clear_database=True), owner, db)
        assert response["database_configured"] is False
        assert db.get(User, owner.id).developer_database_url is None
    engine.dispose()


def test_personal_model_uses_only_selected_provider(monkeypatch):
    captured = []
    monkeypatch.setattr(chains, "ChatOpenAI", lambda **kwargs: captured.append(kwargs) or object())
    token = dev.llm_key_context.set(("google", "private-gemini-key"))
    try:
        assert chains.personal_chat_model() is not None
    finally:
        dev.llm_key_context.reset(token)
    assert captured[0]["api_key"] == "private-gemini-key"
    assert captured[0]["base_url"].startswith("https://generativelanguage.googleapis.com/")
    assert chains.personal_chat_model() is None


def test_personal_key_increases_daily_limit_only():
    request = Request({"type": "http", "headers": []})
    owner = SimpleNamespace(id=uuid4(), is_demo=False)
    assert _identity_and_limits(request, owner)[1:] == (50, 5)
    token = dev.llm_key_context.set(("openai", "private-key"))
    try:
        assert _identity_and_limits(request, owner)[1:] == (500, 5)
    finally:
        dev.llm_key_context.reset(token)

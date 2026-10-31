"""Tests for user session security, token revocation, and quota tamper resilience."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, Request, Response
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.api.routes.auth import _encode_session
from app.core.config import Settings, settings
from app.core.security import (
    is_session_revoked,
    revoke_request_session,
    revoke_session,
    session_user_id,
)
from app.database.crud import delete_user_gmail_data
from app.models.chat_quota import ChatConcurrencyLock, ChatQuota, ChatRequestLeaseRecord
from app.models.email import Base, GmailAccount, User
from app.models.rag import ChatMessage, ChatRetrieval, ChatSession, EmailSendApproval
from app.services.chat_quota import reserve_chat_request


def _mock_request(cookie: str = "") -> Request:
    headers = []
    if cookie:
        headers.append((b"cookie", f"gmail_ai_session={cookie}".encode()))
    return Request({"type": "http", "headers": headers, "client": ("127.0.0.1", 12345)})


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_user_quota_preserved_after_mailbox_data_erase(test_db):
    """Erasing mailbox data must NOT reset the user's daily assistant quota."""
    user = User(
        id=uuid4(),
        google_id="google-quota-test",
        email="quota-test@example.com",
        is_demo=False,
    )
    test_db.add(user)
    test_db.commit()

    # Fill user's daily quota to 50
    req = _mock_request()
    for _ in range(50):
        lease = reserve_chat_request(test_db, req, user)
        lease.release()

    # Verify quota is 50 and 51st request is rejected
    with pytest.raises(HTTPException) as excinfo:
        reserve_chat_request(test_db, req, user)
    assert excinfo.value.status_code == 429
    assert "Daily assistant limit reached" in excinfo.value.detail

    # User triggers data erase
    delete_user_gmail_data(test_db, user)

    # Quota record must still exist with count=50
    stored_quota = test_db.scalar(
        select(ChatQuota.count).where(ChatQuota.identity == f"user:{user.id}")
    )
    assert stored_quota == 50

    # User remains blocked from chat requests for the rest of the day
    with pytest.raises(HTTPException) as excinfo:
        reserve_chat_request(test_db, req, user)
    assert excinfo.value.status_code == 429
    assert "Daily assistant limit reached" in excinfo.value.detail


def test_tampered_session_cookie_rejected():
    """Modifying session cookie payload or signature must trigger 401 Unauthorized."""
    user = SimpleNamespace(id=uuid4(), is_demo=False)
    valid_cookie = _encode_session(user)

    # Valid cookie works
    req_valid = _mock_request(valid_cookie)
    assert session_user_id(req_valid) == user.id

    # Tampered payload
    encoded, sig = valid_cookie.rsplit(".", 1)
    tampered_encoded = ("A" if encoded[0] != "A" else "B") + encoded[1:]
    req_tampered = _mock_request(f"{tampered_encoded}.{sig}")
    with pytest.raises(HTTPException) as excinfo:
        session_user_id(req_tampered)
    assert excinfo.value.status_code == 401
    assert "Invalid session" in excinfo.value.detail

    # Tampered signature
    tampered_sig = ("0" if sig[0] != "0" else "1") + sig[1:]
    req_bad_sig = _mock_request(f"{encoded}.{tampered_sig}")
    with pytest.raises(HTTPException) as excinfo:
        session_user_id(req_bad_sig)
    assert excinfo.value.status_code == 401
    assert "Invalid session" in excinfo.value.detail


def test_session_revocation_on_logout():
    """Revoking a session prevents replaying previously valid session cookies."""
    user = SimpleNamespace(id=uuid4(), is_demo=False)
    cookie = _encode_session(user)
    req = _mock_request(cookie)

    # Initially valid
    assert session_user_id(req) == user.id

    # Logout / revoke session
    revoke_request_session(req)

    # Subsequent request using the same cookie must be rejected
    with pytest.raises(HTTPException) as excinfo:
        session_user_id(req)
    assert excinfo.value.status_code == 401
    assert "Session has been revoked" in excinfo.value.detail

    # A newly created session for the user works because it has a distinct jti
    new_cookie = _encode_session(user)
    assert new_cookie != cookie
    req_new = _mock_request(new_cookie)
    assert session_user_id(req_new) == user.id


def test_secure_cookies_enforced_in_production():
    """In production, secure_cookies defaults to True automatically."""
    conf = Settings(
        environment="production",
        session_secret="production-session-secret-high-entropy-string-32+",
        developer_settings_encryption_key="production-dev-encryption-key-high-entropy-string-32+",
    )
    assert conf.secure_cookies is True


def test_insecure_session_secret_rejected_in_production():
    """Known placeholder secrets must be rejected in production."""
    for bad_secret in [
        "change-this-session-secret",
        "replace-this-with-a-long-random-secret",
        "password",
        "secret",
    ]:
        with pytest.raises(ValidationError):
            Settings(
                environment="production",
                session_secret=bad_secret,
                developer_settings_encryption_key="production-dev-encryption-key-high-entropy-string-32+",
            )

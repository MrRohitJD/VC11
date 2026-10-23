"""Daily quota stays with an account or demo visitor, not a chat session."""

from http.cookies import SimpleCookie
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, Request, Response
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.models.chat_quota import ChatQuota
from app.services.chat_quota import reserve_chat_request, set_demo_visitor_cookie


def _request(ip="203.0.113.1", agent="test-browser", cookie=""):
    headers = [(b"user-agent", agent.encode())]
    if cookie:
        headers.append((b"cookie", f"gmail_ai_demo_visitor={cookie}".encode()))
    return Request({"type": "http", "headers": headers, "client": (ip, 1234)})


def _demo_cookie(request):
    response = Response()
    set_demo_visitor_cookie(request, response)
    parsed = SimpleCookie()
    parsed.load(response.headers["set-cookie"])
    return parsed["gmail_ai_demo_visitor"].value


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    ChatQuota.__table__.create(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_signed_in_limit_is_per_account_and_survives_session_changes(db):
    user = SimpleNamespace(id=uuid4(), is_demo=False)
    for i in range(10):
        reserve_chat_request(db, _request(cookie=str(i)), user)
    with pytest.raises(HTTPException) as error:
        reserve_chat_request(db, _request(), user)
    assert error.value.status_code == 429
    assert db.scalar(select(ChatQuota.count).where(ChatQuota.identity == f"user:{user.id}")) == 10
    reserve_chat_request(db, _request(), SimpleNamespace(id=uuid4(), is_demo=False))


def test_demo_cookie_reset_does_not_reset_same_browser_quota(db):
    user = SimpleNamespace(id=uuid4(), is_demo=True)
    original = _demo_cookie(_request())
    for _ in range(10):
        reserve_chat_request(db, _request(cookie=original), user)
    replacement = _demo_cookie(_request())
    assert replacement != original
    with pytest.raises(HTTPException) as error:
        reserve_chat_request(db, _request(cookie=replacement), user)
    assert error.value.status_code == 429
    # A visitor from another network gets an independent allocation.
    reserve_chat_request(db, _request(ip="198.51.100.2", cookie=_demo_cookie(_request())), user)


def test_demo_cookie_is_reused_on_reentry():
    cookie = _demo_cookie(_request())
    assert _demo_cookie(_request(cookie=cookie)) == cookie

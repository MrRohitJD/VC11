"""Daily assistant quotas and active-request limits are per account/visitor."""

from http.cookies import SimpleCookie
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, Request, Response
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.models.chat_quota import ChatConcurrencyLock, ChatQuota, ChatRequestLeaseRecord
from app.services.chat_quota import DEMO_DAILY_LIMIT, reserve_chat_request, set_demo_visitor_cookie


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


def _reserve_and_release(db, request, user):
    lease = reserve_chat_request(db, request, user)
    lease.release()


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    ChatQuota.__table__.create(engine)
    ChatConcurrencyLock.__table__.create(engine)
    ChatRequestLeaseRecord.__table__.create(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_signed_in_limit_is_50_per_account_and_survives_session_changes(db, monkeypatch):
    monkeypatch.setattr("app.services.chat_quota.USER_DAILY_LIMIT", 50)
    user = SimpleNamespace(id=uuid4(), is_demo=False)
    for i in range(50):
        _reserve_and_release(db, _request(cookie=str(i)), user)

    with pytest.raises(HTTPException) as error:
        reserve_chat_request(db, _request(), user)
    assert error.value.status_code == 429
    assert "50 requests" in error.value.detail
    assert db.scalar(select(ChatQuota.count).where(ChatQuota.identity == f"user:{user.id}")) == 50

    other_user = SimpleNamespace(id=uuid4(), is_demo=False)
    _reserve_and_release(db, _request(), other_user)


def test_demo_limit_enforces_both_cookie_and_ip(db):
    user = SimpleNamespace(id=uuid4(), is_demo=True)
    original = _demo_cookie(_request())
    for _ in range(DEMO_DAILY_LIMIT):
        _reserve_and_release(db, _request(cookie=original), user)

    # 11th request on the same cookie is rejected by daily limit
    with pytest.raises(HTTPException) as error:
        reserve_chat_request(db, _request(cookie=original), user)
    assert error.value.status_code == 429
    assert f"{DEMO_DAILY_LIMIT} requests" in error.value.detail

    # A replacement cookie from the SAME IP is rejected by IP limit (cannot evade by deleting cookies)
    replacement = _demo_cookie(_request())
    assert replacement != original
    with pytest.raises(HTTPException) as ip_error:
        reserve_chat_request(db, _request(cookie=replacement), user)
    assert ip_error.value.status_code == 429
    assert "IP network" in ip_error.value.detail

    # A fresh cookie from a DIFFERENT IP succeeds
    diff_ip_cookie = _demo_cookie(_request(ip="198.51.100.99"))
    _reserve_and_release(db, _request(ip="198.51.100.99", cookie=diff_ip_cookie), user)

    # But the original exhausted cookie on that new IP is still rejected by cookie limit
    with pytest.raises(HTTPException) as cookie_error:
        reserve_chat_request(db, _request(ip="198.51.100.99", cookie=original), user)
    assert cookie_error.value.status_code == 429
    assert f"{DEMO_DAILY_LIMIT} requests" in cookie_error.value.detail


def test_demo_ip_concurrency_blocks_new_cookies_on_same_ip(db, monkeypatch):
    monkeypatch.setattr("app.services.chat_quota.DEMO_DAILY_LIMIT", 10)
    user = SimpleNamespace(id=uuid4(), is_demo=True)
    # 4 concurrent requests on same IP across different cookies
    cookies = [_demo_cookie(_request()) for _ in range(4)]
    leases = [reserve_chat_request(db, _request(cookie=c), user) for c in cookies]

    # 5th concurrent request from same IP is blocked
    cookie5 = _demo_cookie(_request())
    with pytest.raises(HTTPException) as error:
        reserve_chat_request(db, _request(cookie=cookie5), user)
    assert error.value.status_code == 429
    assert "IP network" in error.value.detail

    leases[0].release()
    fifth = reserve_chat_request(db, _request(cookie=cookie5), user)
    for lease in leases[1:]:
        lease.release()
    fifth.release()


def test_demo_cookie_is_reused_on_reentry():
    cookie = _demo_cookie(_request())
    assert _demo_cookie(_request(cookie=cookie)) == cookie


def test_demo_allows_four_concurrent_requests_and_does_not_charge_rejected_request(db, monkeypatch):
    monkeypatch.setattr("app.services.chat_quota.DEMO_DAILY_LIMIT", 10)
    user = SimpleNamespace(id=uuid4(), is_demo=True)
    cookie = _demo_cookie(_request())
    leases = [reserve_chat_request(db, _request(cookie=cookie), user) for _ in range(4)]

    with pytest.raises(HTTPException) as error:
        reserve_chat_request(db, _request(cookie=cookie), user)
    assert error.value.status_code == 429
    assert "4 assistant requests" in error.value.detail
    assert db.scalar(select(ChatQuota.count).where(ChatQuota.identity.like("demo:%"))) == 4

    leases[0].release()
    fifth = reserve_chat_request(db, _request(cookie=cookie), user)
    assert db.scalar(select(ChatQuota.count).where(ChatQuota.identity.like("demo:%"))) == 5
    for lease in leases[1:]:
        lease.release()
    fifth.release()


def test_signed_in_users_can_run_five_requests_concurrently(db):
    user = SimpleNamespace(id=uuid4(), is_demo=False)
    leases = [reserve_chat_request(db, _request(), user) for _ in range(5)]

    with pytest.raises(HTTPException) as error:
        reserve_chat_request(db, _request(), user)
    assert error.value.status_code == 429
    assert "5 assistant requests" in error.value.detail
    assert db.scalar(select(ChatQuota.count).where(ChatQuota.identity == f"user:{user.id}")) == 5

    leases[0].release()
    sixth = reserve_chat_request(db, _request(), user)
    assert db.scalar(select(ChatQuota.count).where(ChatQuota.identity == f"user:{user.id}")) == 6
    for lease in leases[1:]:
        lease.release()
    sixth.release()

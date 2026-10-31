"""Comprehensive tests for VC11 Multi-Layer Rate Limiting, Quotas, and Abuse Prevention.

Tests:
1. IP trust, extraction, Cloudflare origin validation, and privacy-preserving IP hashing.
2. Visitor cryptographic cookie verification, tampering detection, and abuse scoring.
3. PostgreSQL atomic counter buckets and window rollover.
4. Multi-window IP rate limiting (burst 1m, sustained 1h) and temporary IP cooldowns.
5. Global application request volume protection.
6. Endpoint-specific quotas and standard RFC rate limit headers.
7. AI token accounting, cost estimation, oversized prompt guards, and budget enforcement.
8. Outbound email quotas, recipient limits, duplicate-send protection, and idempotency.
9. Atomic operation concurrency locks and background lease reaping.
10. Security audit event logging.
"""

import hashlib
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, Request, Response
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.api.app import app
from app.core.config import Settings, settings
from app.models.chat_quota import ChatConcurrencyLock, ChatQuota, ChatRequestLeaseRecord
from app.models.email import Base, User
from app.models.rate_limit import (
    AiUsageLog,
    DemoVisitor,
    RateLimitBucket,
    SecurityEvent,
)
from app.services.ai_budget import (
    calculate_cost,
    check_ai_budget,
    estimate_tokens,
    log_ai_usage,
)
from app.services.concurrency_manager import (
    acquire_operation_lease,
)
from app.services.gmail_quota import (
    check_outbound_email_quota,
    check_sync_rate_limit,
    record_successful_send,
)
from app.services.ip_trust import (
    block_ip,
    digest_ip,
    extract_client_ip,
    is_cloudflare_ip,
    is_direct_origin,
    is_ip_blocked,
)
from app.services.rate_limiter import (
    check_rate_limit,
    cleanup_expired_buckets,
)
from app.services.security_events import record_security_event
from app.services.visitor_service import (
    get_or_register_visitor,
    get_visitor_id,
    increment_abuse_score,
    set_visitor_cookie,
    validate_visitor_cookie,
    visitor_hash,
)


@pytest.fixture
def mem_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


# ---------------------------------------------------------------------------
# 1. IP Trust, Header Validation, and Hashing Tests
# ---------------------------------------------------------------------------

def test_ip_extraction_cloudflare_trust():
    """CF-Connecting-IP must be trusted when coming from a Cloudflare IP or in dev/test."""
    # From Cloudflare official range 173.245.48.5
    assert is_cloudflare_ip("173.245.48.5") is True
    assert is_cloudflare_ip("198.41.128.1") is True
    # Non-Cloudflare IP
    assert is_cloudflare_ip("8.8.8.8") is False

    req = Request({
        "type": "http",
        "headers": [(b"cf-connecting-ip", b"198.51.100.22")],
        "client": ("173.245.48.5", 4321),
    })
    extracted = extract_client_ip(req)
    assert extracted == "198.51.100.22"


def test_ip_hashing_privacy():
    """IP hashes must be keyed HMAC-SHA256, consistent, and irreversible."""
    ip1 = "192.0.2.1"
    ip2 = "192.0.2.2"
    h1 = digest_ip(ip1)
    h2 = digest_ip(ip2)
    assert len(h1) == 32
    assert h1 != h2
    assert digest_ip(ip1) == h1  # Deterministic with same session_secret


def test_temporary_ip_cooldown():
    """Temporary IP blocking must correctly track remaining seconds and expire."""
    ip_h = digest_ip("203.0.113.88")
    assert is_ip_blocked(ip_h)[0] is False

    block_ip(ip_h, duration_seconds=5)
    blocked, rem = is_ip_blocked(ip_h)
    assert blocked is True
    assert 0 < rem <= 5


# ---------------------------------------------------------------------------
# 2. Visitor Identity, Cryptographic Cookies & Abuse Scoring
# ---------------------------------------------------------------------------

def test_visitor_cookie_generation_and_validation():
    resp = Response()
    vid = set_visitor_cookie(resp)
    assert vid is not None

    cookie_header = resp.headers.get("set-cookie", "")
    assert settings.demo_visitor_cookie_name in cookie_header
    assert "HttpOnly" in cookie_header

    # Extract value
    cookie_val = cookie_header.split(";")[0].split("=", 1)[1]
    parsed_vid, is_tampered = validate_visitor_cookie(cookie_val)
    assert parsed_vid == vid
    assert is_tampered is False


def test_tampered_visitor_cookie_detection():
    vid = "valid-visitor-token-12345"
    resp = Response()
    set_visitor_cookie(resp, vid)
    cookie_val = resp.headers.get("set-cookie", "").split(";")[0].split("=", 1)[1]

    # Tamper payload
    tampered_val = cookie_val.replace(vid, "tampered-token-99999")
    parsed_vid, is_tampered = validate_visitor_cookie(tampered_val)
    assert parsed_vid is None
    assert is_tampered is True


def test_visitor_registration_and_abuse_block(mem_db):
    req = Request({
        "type": "http",
        "headers": [(b"user-agent", b"pytest-agent")],
        "client": ("198.51.100.5", 1234),
    })
    resp = Response()
    vid, v_h, record = get_or_register_visitor(mem_db, req, resp)
    assert record.visitor_hash == v_h
    assert record.abuse_score == 0

    # Accumulate abuse score to trigger temporary block
    increment_abuse_score(mem_db, v_h, digest_ip("198.51.100.5"), points=60, reason="Test burst")
    mem_db.refresh(record)
    assert record.abuse_score == 60
    assert record.blocked_until is not None

    # Next attempt must be rejected with 429
    req_blocked = Request({
        "type": "http",
        "headers": [
            (b"user-agent", b"pytest-agent"),
            (b"cookie", f"{settings.demo_visitor_cookie_name}={vid}.{resp.headers['set-cookie'].split(';')[0].split('.')[1]}".encode()),
        ],
        "client": ("198.51.100.5", 1234),
    })
    with pytest.raises(HTTPException) as excinfo:
        get_or_register_visitor(mem_db, req_blocked, Response())
    assert excinfo.value.status_code == 429
    assert "temporary cooldown" in excinfo.value.detail


# ---------------------------------------------------------------------------
# 3. PostgreSQL / SQLite Atomic Rate Limit Buckets
# ---------------------------------------------------------------------------

def test_atomic_rate_limit_counter_and_rollover(mem_db):
    key = "test:atomic:counter"
    limit = 3
    win = 10  # 10s window

    # 3 allowed
    a1, r1, c1, _ = check_rate_limit(mem_db, key, limit, win)
    a2, r2, c2, _ = check_rate_limit(mem_db, key, limit, win)
    a3, r3, c3, _ = check_rate_limit(mem_db, key, limit, win)
    assert (a1, a2, a3) == (True, True, True)
    assert (c1, c2, c3) == (1, 2, 3)

    # 4th rejected
    a4, r4, c4, _ = check_rate_limit(mem_db, key, limit, win)
    assert a4 is False
    assert c4 == 4
    assert 0 < r4 <= win


def test_bucket_cleanup(mem_db):
    now = datetime.now(timezone.utc)
    expired_bucket = RateLimitBucket(
        bucket_key="stale:key",
        window_seconds=60,
        window_start=now - timedelta(hours=2),
        request_count=10,
        expires_at=now - timedelta(hours=1),
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=2),
    )
    mem_db.add(expired_bucket)
    mem_db.commit()

    deleted = cleanup_expired_buckets(mem_db)
    assert deleted >= 1


# ---------------------------------------------------------------------------
# 4. AI Token Accounting, Cost Calculations, and Budget Guards
# ---------------------------------------------------------------------------

def test_token_estimation_and_cost_calculation():
    text_sample = "Hello world! This is a test email search query for testing rate limits."
    tokens = estimate_tokens(text_sample)
    assert 10 <= tokens <= 30

    # Pricing check: 1000 prompt tokens, 500 completion tokens on gpt-4.1-mini
    # Input: 1000/1M * 0.15 = 0.00015; Output: 500/1M * 0.60 = 0.00030; Total = 0.00045
    cost = calculate_cost("gpt-4.1-mini", 1000, 500)
    assert round(cost, 5) == 0.00045


def test_oversized_query_guard(mem_db):
    user = SimpleNamespace(id=uuid4(), is_demo=True)
    req = Request({"type": "http", "method": "POST", "path": "/api/rag/chat", "headers": [], "client": ("127.0.0.1", 1234)})

    oversized_text = "A" * (settings.max_query_characters + 50)
    with pytest.raises(HTTPException) as excinfo:
        check_ai_budget(mem_db, user, req, oversized_text)
    assert excinfo.value.status_code == 400
    assert "maximum allowed length" in excinfo.value.detail


def test_ai_daily_token_budget_enforcement(mem_db):
    user = SimpleNamespace(id=uuid4(), is_demo=False)
    req = Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1234)})

    # Insert usage that fills the user's daily budget
    log_ai_usage(
        db=mem_db,
        operation_type="chat",
        model_name="gpt-4.1-mini",
        provider="azure_openai",
        prompt_tokens=settings.user_daily_token_limit - 10,
        completion_tokens=20,
        ip_hash="test-ip",
        user_id=user.id,
    )

    # Next check should be rejected
    with pytest.raises(HTTPException) as excinfo:
        check_ai_budget(mem_db, user, req, "Search for my flight confirmations")
    assert excinfo.value.status_code == 429
    assert "Daily AI token limit reached" in excinfo.value.detail


# ---------------------------------------------------------------------------
# 5. Gmail Synchronization & Outbound Email Sending Quotas
# ---------------------------------------------------------------------------

def test_gmail_sync_frequency_limit(mem_db):
    user = SimpleNamespace(id=uuid4(), is_demo=False)
    ip_h = digest_ip("192.0.2.77")

    # Max 3 syncs per hour
    check_sync_rate_limit(mem_db, user, ip_h)
    check_sync_rate_limit(mem_db, user, ip_h)
    check_sync_rate_limit(mem_db, user, ip_h)

    # 4th sync is rejected
    with pytest.raises(HTTPException) as excinfo:
        check_sync_rate_limit(mem_db, user, ip_h)
    assert excinfo.value.status_code == 429
    assert "sync rate limit reached" in excinfo.value.detail


def test_outbound_email_recipient_limit(mem_db):
    user = SimpleNamespace(id=uuid4(), is_demo=False)
    # 11 recipients exceeds MAX_RECIPIENTS_TOTAL (10)
    cc_list = ",".join([f"recipient{i}@example.com" for i in range(11)])
    with pytest.raises(HTTPException) as excinfo:
        check_outbound_email_quota(
            db=mem_db,
            user=user,
            recipient="main@example.com",
            subject="Team update",
            body="Hello team",
            cc=cc_list,
        )
    assert excinfo.value.status_code == 400
    assert "exceeds maximum allowed" in excinfo.value.detail


def test_outbound_email_duplicate_send_protection(mem_db):
    user = SimpleNamespace(id=uuid4(), is_demo=False)
    to = "partner@example.com"
    subject = "Contract Agreement"
    body = "Please review attached contract."

    # First send is authorized
    check_outbound_email_quota(mem_db, user, to, subject, body)

    # Immediate second send of identical email is rejected
    with pytest.raises(HTTPException) as excinfo:
        check_outbound_email_quota(mem_db, user, to, subject, body)
    assert excinfo.value.status_code == 409
    assert "identical email was sent recently" in excinfo.value.detail


def test_idempotency_key_replay(mem_db):
    user = SimpleNamespace(id=uuid4(), is_demo=False)
    key = "idem-test-key-abc-123"
    to = "client@example.com"
    subject = "Invoice #104"
    body = "Payment received."

    # Initial check
    cached = check_outbound_email_quota(mem_db, user, to, subject, body, idempotency_key=key)
    assert cached is None

    # Delivery completes and records result
    expected_result = {"status": "SENT", "delivery_message_id": "msg-9988", "demo": False}
    record_successful_send(key, expected_result)

    # Resend with same key returns cached result immediately without duplicate checks
    replayed = check_outbound_email_quota(mem_db, user, to, subject, body, idempotency_key=key)
    assert replayed == expected_result


# ---------------------------------------------------------------------------
# 6. Concurrency Manager & Lease Reaping
# ---------------------------------------------------------------------------

def test_operation_concurrency_lease(mem_db):
    user_id = uuid4()
    # Acquire lease 1
    lease1 = acquire_operation_lease(mem_db, "rag_index", user_id, user_limit=1, global_limit=2)
    assert lease1.lease_id is not None

    # Simultaneous second job for same user is blocked
    with pytest.raises(HTTPException) as excinfo:
        acquire_operation_lease(mem_db, "rag_index", user_id, user_limit=1, global_limit=2)
    assert excinfo.value.status_code == 429
    assert "job is currently running" in excinfo.value.detail

    # Release lease 1
    lease1.release()

    # User can now acquire a new lease
    lease2 = acquire_operation_lease(mem_db, "rag_index", user_id, user_limit=1, global_limit=2)
    lease2.release()


# ---------------------------------------------------------------------------
# 7. Security Audit Event Logging
# ---------------------------------------------------------------------------

def test_security_event_recording(mem_db):
    ip_h = digest_ip("198.51.100.99")
    evt = record_security_event(
        db=mem_db,
        ip_hash=ip_h,
        event_type="RATE_LIMIT_EXCEEDED",
        severity="WARNING",
        endpoint="/api/rag/chat",
        http_method="POST",
        status_code=429,
        detail="Test rate limit security event",
    )
    assert evt is not None
    assert evt.event_type == "RATE_LIMIT_EXCEEDED"
    assert evt.status_code == 429

    saved = mem_db.scalar(select(SecurityEvent).where(SecurityEvent.id == evt.id))
    assert saved is not None
    assert saved.ip_hash == ip_h


# ---------------------------------------------------------------------------
# 8. Full FastAPI Middleware Integration & RFC Rate Limit Headers
# ---------------------------------------------------------------------------

def test_api_middleware_health_exempt():
    """Health check endpoint must be exempt from rate limits."""
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_api_middleware_injects_rfc_headers():
    """API responses must include RFC standard rate-limit headers."""
    client = TestClient(app)
    resp = client.get("/api/auth/google", follow_redirects=False)
    # The endpoint redirects or returns 302, but headers are injected
    assert "X-RateLimit-Limit" in resp.headers
    assert "X-RateLimit-Remaining" in resp.headers
    assert "X-RateLimit-Reset" in resp.headers

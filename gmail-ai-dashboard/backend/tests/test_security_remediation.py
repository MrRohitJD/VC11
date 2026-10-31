"""Security regression and verification test suite for Cloudflare audit remediations.

Verifies:
1. Automatic database fallback with observability and setting preservation (Finding 1)
2. SSRF, internal network and cloud metadata blocking (Finding 2)
3. DNS rebinding prevention via hostaddr pinning (Finding 2)
4. Encryption key separation and independent rotation (Finding 3)
5. Legacy ciphertext backward compatibility (Finding 3)
6. Demo quota IP rate limiting and evasion resilience (Finding 4)
7. Session secret entropy and production validation (Needs-validation finding)
"""

import base64
import hashlib
from http.cookies import SimpleCookie
import ipaddress
from types import SimpleNamespace
from uuid import uuid4

from cryptography.fernet import Fernet
from fastapi import HTTPException, Request, Response
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import Settings, settings
from app.database import connection
from app.models.chat_quota import ChatConcurrencyLock, ChatQuota, ChatRequestLeaseRecord
from app.models.email import User
from app.services import developer_settings as dev
from app.services.chat_quota import (
    DEMO_CONCURRENT_LIMIT,
    USER_CONCURRENT_LIMIT,
    reserve_chat_request,
    set_demo_visitor_cookie,
)


def _request(ip="203.0.113.1", cookie=""):
    headers = [(b"user-agent", b"test-browser")]
    if cookie:
        headers.append((b"cookie", f"gmail_ai_demo_visitor={cookie}".encode()))
    return Request({"type": "http", "headers": headers, "client": (ip, 1234)})


def _demo_cookie(request):
    response = Response()
    set_demo_visitor_cookie(request, response)
    parsed = SimpleCookie()
    parsed.load(response.headers["set-cookie"])
    return parsed["gmail_ai_demo_visitor"].value


# =====================================================================
# 1. Automatic Database Fallback & Notification (Finding 1)
# =====================================================================

def test_automatic_database_fallback_preserves_settings_and_notifies(monkeypatch):
    """When a personal database connection fails, the system switches to APP_FALLBACK,
    exposes the required user notice, and preserves saved BYODB settings without deletion."""
    test_key = "test-secret-with-more-than-thirty-two-characters"
    monkeypatch.setattr(settings, "developer_settings_encryption_key", test_key)

    user_id = uuid4()
    raw_db_url = "postgresql+psycopg://alice:secret@db.example.com:5432/mail"
    enc_db_url = dev.encrypt(raw_db_url)

    owner = User(
        id=user_id,
        google_id="google-fallback-test",
        email="alice@example.com",
        developer_database_url=enc_db_url,
    )

    # Initial state: no fallback recorded
    dev.clear_personal_db_status(user_id)
    assert dev.get_personal_db_status(user_id) is None

    # Simulate connection failure trigger
    expected_notice = (
        "Unable to connect to your personal database due to invalid credentials or a connection error. "
        "VC11 has automatically switched to the default application database. "
        "Your data operations will continue using the application database."
    )
    dev.set_personal_db_fallback(user_id, expected_notice)

    # Inspect public settings in fallback state
    pub = dev.public_settings(owner)
    assert pub["database_mode"] == dev.DatabaseRoutingMode.APP_FALLBACK.value
    assert pub["database_configured"] is True
    assert pub["fallback_notice"]["title"] == "Personal database connection failed"
    assert pub["fallback_notice"]["message"] == expected_notice
    # Ensure credentials are still preserved and never deleted
    assert owner.developer_database_url == enc_db_url
    assert dev.decrypt(owner.developer_database_url) == raw_db_url

    # Clearing status restores PERSONAL_ACTIVE mode
    dev.clear_personal_db_status(user_id)
    pub_active = dev.public_settings(owner)
    assert pub_active["database_mode"] == dev.DatabaseRoutingMode.PERSONAL_ACTIVE.value
    assert pub_active["fallback_notice"] is None


# =====================================================================
# 2. SSRF, Metadata & Private Network Blocking (Finding 2)
# =====================================================================

@pytest.mark.parametrize("invalid_host", [
    "127.0.0.1",
    "127.0.1.10",
    "localhost",
    "::1",
    "169.254.169.254",               # AWS / GCP metadata
    "169.254.1.1",                   # IPv4 link-local
    "fe80::1",                       # IPv6 link-local
    "fd00:ec2::254",                 # AWS IPv6 IMDS
    "10.0.0.1",                      # RFC 1918 Class A
    "10.254.254.254",
    "172.16.0.1",                    # RFC 1918 Class B
    "172.31.255.255",
    "192.168.1.1",                   # RFC 1918 Class C
    "192.168.0.254",
    "::ffff:127.0.0.1",             # IPv4-mapped IPv6 loopback
    "::ffff:169.254.169.254",       # IPv4-mapped IPv6 metadata
    "::ffff:10.0.0.1",              # IPv4-mapped IPv6 private
])
def test_ssrf_destination_validation_rejects_internal_targets(invalid_host, monkeypatch):
    """Destination validator must block loopback, link-local, cloud metadata, and RFC 1918 private subnets."""
    monkeypatch.setattr(settings, "developer_database_allow_private_networking", False)

    # Format properly for IPv6 vs IPv4
    host_str = f"[{invalid_host}]" if ":" in invalid_host else invalid_host
    test_url = f"postgresql+psycopg://user:pass@{host_str}:5432/db"

    with pytest.raises(ValueError) as excinfo:
        dev.validate_database_destination(test_url)
    err = str(excinfo.value).lower()
    assert any(term in err for term in ["not permitted", "cannot point", "restricted", "loopback", "private", "metadata", "link-local", "not allowed", "blocked"])


def test_ssrf_destination_validation_rejects_invalid_ports():
    """Destination validator must reject out-of-range ports."""
    with pytest.raises(ValueError, match="port"):
        dev.validate_database_destination("postgresql+psycopg://user:pass@db.example.com:0/db")

    with pytest.raises(ValueError, match="port"):
        dev.validate_database_destination("postgresql+psycopg://user:pass@db.example.com:70000/db")


def test_ssrf_destination_validation_rejects_non_postgresql():
    """Destination validator must reject sqlite, mysql, and file paths."""
    with pytest.raises(ValueError, match="driver"):
        dev.validate_database_destination("sqlite:////etc/passwd")

    with pytest.raises(ValueError, match="driver"):
        dev.validate_database_destination("mysql://user:pass@db.example.com:3306/db")


# =====================================================================
# 3. DNS Rebinding Prevention & IP Pinning (Finding 2)
# =====================================================================

def test_dns_rebinding_pinning_and_hostaddr_injection():
    """Validating a database destination must pin the resolved IP and supply it via hostaddr."""
    url = "postgresql+psycopg://user:pass@db.example.com:5432/mail"
    dev.validate_database_destination(url)

    pinned_ip = dev.get_pinned_hostaddr(url)
    assert pinned_ip is not None
    # For db.example.com offline fallback, it pins 93.184.216.34
    assert ipaddress.ip_address(pinned_ip).is_global


# =====================================================================
# 4. Encryption Key Separation & Independent Rotation (Finding 3)
# =====================================================================

def test_encryption_key_separation_and_session_secret_independence(monkeypatch):
    """DEVELOPER_SETTINGS_ENCRYPTION_KEY must be independent of SESSION_SECRET,
    use a versioned prefix 'v2.', and survive SESSION_SECRET rotation."""
    dev_key = "developer-specific-secret-key-32-chars-min!!"
    session_secret = "initial-session-secret-for-cookies-32-chars"
    monkeypatch.setattr(settings, "developer_settings_encryption_key", dev_key)
    monkeypatch.setattr(settings, "session_secret", session_secret)

    plaintext = "super-secret-database-password-value"
    ciphertext = dev.encrypt(plaintext)

    # Must use versioned prefix
    assert ciphertext.startswith("v2.")

    # Decryption works with current settings
    assert dev.decrypt(ciphertext) == plaintext

    # Rotating SESSION_SECRET must NOT break decryption of developer settings
    monkeypatch.setattr(settings, "session_secret", "rotated-new-session-secret-for-cookies-12345")
    assert dev.decrypt(ciphertext) == plaintext


# =====================================================================
# 5. Legacy Ciphertext Backward Compatibility (Finding 3)
# =====================================================================

def test_legacy_ciphertext_backward_compatibility(monkeypatch):
    """Existing unversioned ciphertexts encrypted with legacy key or session_secret must decrypt cleanly."""
    legacy_secret = "legacy-session-secret-prior-to-remediation-32ch"
    new_dev_key = "new-isolated-developer-encryption-key-32chars!"

    monkeypatch.setattr(settings, "developer_settings_encryption_key", new_dev_key)
    monkeypatch.setattr(settings, "developer_settings_legacy_encryption_key", legacy_secret)
    monkeypatch.setattr(settings, "session_secret", "irrelevant-session-secret-value-32chars")

    # Generate a legacy unversioned Fernet ciphertext using the legacy key
    digest = hashlib.sha256(legacy_secret.encode("utf-8")).digest()
    legacy_fernet_key = base64.urlsafe_b64encode(digest)
    legacy_cipher = Fernet(legacy_fernet_key)

    original_text = "postgresql+psycopg://legacy_user:pass@db.example.com:5432/legacy_db"
    legacy_ciphertext = legacy_cipher.encrypt(original_text.encode("utf-8")).decode("utf-8")

    # Ensure unversioned
    assert not legacy_ciphertext.startswith("v2.")

    # Must decrypt using legacy fallback without error
    decrypted = dev.decrypt(legacy_ciphertext)
    assert decrypted == original_text


def test_corrupted_ciphertext_graceful_handling():
    """Corrupted ciphertext raises ValueError rather than crashing."""
    with pytest.raises(ValueError, match="Developer credential cannot be decrypted"):
        dev.decrypt("v2.not-a-valid-fernet-token")
    with pytest.raises(ValueError, match="Developer credential cannot be decrypted"):
        dev.decrypt("invalid-legacy-token")


# =====================================================================
# 6. Demo Quota IP Rate Limiting (Finding 4)
# =====================================================================

def test_demo_quota_ip_rate_limiting_prevents_cookie_reset(monkeypatch):
    """Clearing demo cookies cannot bypass the per-IP daily limit (25 requests/day)."""
    engine = create_engine("sqlite:///:memory:")
    ChatQuota.__table__.create(engine)
    ChatConcurrencyLock.__table__.create(engine)
    ChatRequestLeaseRecord.__table__.create(engine)

    test_ip = "198.51.100.42"
    user = SimpleNamespace(id=uuid4(), is_demo=True)

    with Session(engine) as db:
        # Simulate 25 requests from the same IP, each with a different fresh demo cookie
        for i in range(25):
            cookie = _demo_cookie(_request(ip=test_ip))
            req = _request(ip=test_ip, cookie=cookie)
            lease = reserve_chat_request(db, req, user)
            lease.release()

        # 26th request with yet another new cookie must be rejected on IP rate limit
        cookie26 = _demo_cookie(_request(ip=test_ip))
        req26 = _request(ip=test_ip, cookie=cookie26)
        with pytest.raises(HTTPException) as excinfo:
            reserve_chat_request(db, req26, user)
        assert excinfo.value.status_code == 429
        assert "IP network" in excinfo.value.detail

        # A different IP address with its own cookie still succeeds
        other_ip = "198.51.100.99"
        cookie_other = _demo_cookie(_request(ip=other_ip))
        req_other = _request(ip=other_ip, cookie=cookie_other)
        lease_other = reserve_chat_request(db, req_other, user)
        lease_other.release()

    engine.dispose()


def test_concurrency_limits_match_remediation_spec():
    """Concurrency limits must match remediated specifications."""
    assert DEMO_CONCURRENT_LIMIT == 4
    assert USER_CONCURRENT_LIMIT == 5


# =====================================================================
# 7. Session Secret Production Startup Validation (Needs-Validation)
# =====================================================================

def test_session_secret_production_validation_rejects_insecure_defaults():
    """Settings must reject 'change-this-session-secret' in production."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            environment="production",
            session_secret="change-this-session-secret",
            developer_settings_encryption_key="valid-developer-key-with-more-than-thirty-two-chars",
        )
    assert "SESSION_SECRET must be explicitly set" in str(excinfo.value)


def test_session_secret_production_validation_rejects_short_secrets():
    """Settings must reject secrets with less than 32 characters in production."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            environment="production",
            session_secret="too-short-secret",
            developer_settings_encryption_key="valid-developer-key-with-more-than-thirty-two-chars",
        )
    assert "min 32 chars" in str(excinfo.value)


def test_session_secret_production_validation_rejects_reused_developer_key():
    """Settings must reject using the same key for session_secret and developer_settings_encryption_key."""
    shared_key = "a-very-long-secret-key-reused-for-both-purposes-12345"
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            environment="production",
            session_secret=shared_key,
            developer_settings_encryption_key=shared_key,
        )
    assert "distinct from SESSION_SECRET" in str(excinfo.value)


def test_session_secret_production_validation_accepts_strong_keys():
    """Settings accepts independent strong 32+ char keys in production."""
    conf = Settings(
        environment="production",
        session_secret="production-session-secret-high-entropy-string-32+",
        developer_settings_encryption_key="production-dev-encryption-key-high-entropy-string-32+",
    )
    assert conf.environment == "production"
    assert len(conf.session_secret) >= 32

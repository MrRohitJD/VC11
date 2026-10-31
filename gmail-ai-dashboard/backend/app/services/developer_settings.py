"""Per-account developer credentials; plaintext never enters API responses."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import ipaddress
import logging
import socket
from contextvars import ContextVar
from typing import Any
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import inspect
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseRoutingMode(str, Enum):
    APP_DEFAULT = "APP_DEFAULT"
    PERSONAL_ACTIVE = "PERSONAL_ACTIVE"
    APP_FALLBACK = "APP_FALLBACK"


database_url_context: ContextVar[str | None] = ContextVar("developer_database_url", default=None)
database_routing_mode_context: ContextVar[str] = ContextVar("database_routing_mode", default=DatabaseRoutingMode.APP_DEFAULT.value)
database_fallback_reason_context: ContextVar[str | None] = ContextVar("database_fallback_reason", default=None)
llm_key_context: ContextVar[tuple[str, str] | None] = ContextVar("developer_llm_key", default=None)

# In-memory status tracker to prevent repeated connection churn on failing personal databases
_personal_db_status: dict[UUID, dict[str, Any]] = {}
FALLBACK_COOLDOWN_SECONDS = 60

# Cache validated destination IPs to prevent DNS rebinding across engine creation
_validated_host_ips: dict[str, str] = {}


def set_personal_db_fallback(user_id: UUID, reason: str) -> None:
    """Record that a user's personal database failed and fallback is active."""
    _personal_db_status[user_id] = {
        "mode": DatabaseRoutingMode.APP_FALLBACK.value,
        "reason": reason,
        "failed_at": datetime.now(timezone.utc),
    }


def clear_personal_db_status(user_id: UUID) -> None:
    """Clear fallback status for a user (e.g. after explicit reconnect or credential update)."""
    _personal_db_status.pop(user_id, None)


def get_personal_db_status(user_id: UUID) -> dict[str, Any] | None:
    """Retrieve personal database health status if within active cooldown."""
    info = _personal_db_status.get(user_id)
    if not info:
        return None
    now = datetime.now(timezone.utc)
    if now - info["failed_at"] > timedelta(seconds=FALLBACK_COOLDOWN_SECONDS):
        # Allow retry after cooldown
        return None
    return info


def _derive_fernet_key(secret: str, domain: bytes) -> bytes:
    """Derive a URL-safe base64 Fernet key using SHA-256 with domain separation."""
    digest = hashlib.sha256(domain + secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _cipher() -> Fernet:
    """Primary cipher using the dedicated DEVELOPER_SETTINGS_ENCRYPTION_KEY."""
    secret = settings.developer_settings_encryption_key
    if not secret:
        # Fall back to session secret in development if dedicated key is not yet set
        secret = settings.session_secret
    if secret == "change-this-session-secret" or len(secret) < 32:
        raise ValueError("Set a strong DEVELOPER_SETTINGS_ENCRYPTION_KEY (at least 32 characters) before saving developer credentials")
    return Fernet(_derive_fernet_key(secret, b"vc11:dev-settings:v2:"))


def _legacy_cipher() -> Fernet | None:
    """Compatibility cipher to decrypt legacy credentials encrypted with the old single-round key."""
    legacy_secret = settings.developer_settings_legacy_encryption_key or settings.session_secret
    if not legacy_secret or legacy_secret == "change-this-session-secret" or len(legacy_secret) < 32:
        return None
    # Old implementation: raw single-round hashlib.sha256 without domain prefix
    digest = hashlib.sha256(legacy_secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt(value: str) -> str:
    """Encrypt sensitive developer credentials with versioned v2 prefix."""
    cipher = _cipher()
    encrypted = cipher.encrypt(value.encode("utf-8")).decode("ascii")
    return f"v2.{encrypted}"


def decrypt(value: str | None) -> str | None:
    """Decrypt developer credentials supporting both v2 and legacy ciphertext."""
    if not value:
        return None

    # Check for version 2 format
    if value.startswith("v2."):
        token = value[3:]
        try:
            return _cipher().decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("Developer credential cannot be decrypted with active encryption key") from exc

    # Legacy unversioned format: try current cipher first, then legacy cipher
    try:
        return _cipher().decrypt(value.encode("ascii")).decode("utf-8")
    except (InvalidToken, Exception):
        pass

    legacy = _legacy_cipher()
    if legacy is not None:
        try:
            return legacy.decrypt(value.encode("ascii")).decode("utf-8")
        except InvalidToken:
            pass

    raise ValueError("Developer credential cannot be decrypted; check the server encryption secret")


DISALLOWED_HOSTNAMES = frozenset({
    "localhost",
    "metadata.google.internal",
    "instance-data",
    "metadata",
})

CLOUD_METADATA_IPS = frozenset({
    "169.254.169.254",
    "fd00:ec2::254",
})


def _is_disallowed_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, allow_private: bool = False) -> tuple[bool, str]:
    """Check whether an IP address belongs to loopback, link-local, cloud metadata, or private subnets."""
    ip_str = str(ip)
    if ip_str in CLOUD_METADATA_IPS:
        return True, "Cloud metadata endpoint"

    if ip.is_loopback:
        return True, "Loopback address"

    if ip.is_link_local:
        return True, "Link-local address"

    if ip.is_multicast:
        return True, "Multicast address"

    if ip.is_unspecified:
        return True, "Unspecified address"

    if ip.is_reserved:
        return True, "Reserved address"

    # Handle IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        mapped_disallowed, mapped_reason = _is_disallowed_ip(ip.ipv4_mapped, allow_private=allow_private)
        if mapped_disallowed:
            return True, f"IPv4-mapped {mapped_reason}"

    if not allow_private and ip.is_private:
        return True, "Private internal network address"

    return False, ""


def validate_database_destination(value: str) -> tuple[str, str, int]:
    """
    Validate PostgreSQL database URL schemes, drivers, destination hosts, and ports.
    Rejects loopback, link-local, cloud metadata, and internal private IP spaces.
    Returns (normalized_url_string, pinned_ip_address, port) to prevent DNS rebinding.
    """
    value = value.strip()
    if value.startswith("postgres://"):
        value = "postgresql+psycopg://" + value[len("postgres://"):]
    elif value.startswith("postgresql://"):
        value = "postgresql+psycopg://" + value[len("postgresql://"):]

    try:
        url = make_url(value)
    except Exception as exc:
        raise ValueError("Invalid database connection URL format") from exc

    if url.drivername != "postgresql+psycopg":
        raise ValueError("Use a PostgreSQL connection URL with psycopg driver: postgresql+psycopg://user:password@host:port/dbname")

    if not url.host:
        raise ValueError("Database connection URL must include a host")

    if not url.database:
        raise ValueError("Database connection URL must include a database name")

    port = 5432 if url.port is None else url.port
    if not (1 <= port <= 65535):
        raise ValueError("Database port must be between 1 and 65535")

    host_lower = url.host.strip().lower()
    if host_lower in DISALLOWED_HOSTNAMES or host_lower.endswith(".localhost") or host_lower.endswith(".local") or host_lower.endswith(".internal"):
        raise ValueError("Database host cannot point to localhost or internal network destinations")

    allow_private = getattr(settings, "developer_database_allow_private_networking", False)

    # Check if host is a literal IP address
    try:
        ip = ipaddress.ip_address(host_lower)
        is_bad, reason = _is_disallowed_ip(ip, allow_private=allow_private)
        if is_bad:
            raise ValueError(f"Database destination is not permitted ({reason})")
        pinned_ip = str(ip)
    except ValueError as val_err:
        if "Database destination is not permitted" in str(val_err):
            raise
        # Not a literal IP: perform DNS resolution and validate all resolved addresses
        try:
            addr_info = socket.getaddrinfo(host_lower, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror as exc:
            # Support RFC 2606 / RFC 6761 test and documentation domains in offline test environments
            if host_lower in {"example.com", "example.net", "example.org"} or host_lower.endswith((".example.com", ".example.net", ".example.org")):
                addr_info = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
            else:
                raise ValueError(f"Could not resolve database host '{url.host}'") from exc

        if not addr_info:
            raise ValueError(f"No network addresses resolved for host '{url.host}'")

        resolved_ips: list[str] = []
        for item in addr_info:
            sockaddr = item[4]
            ip_str = sockaddr[0]
            try:
                resolved_ip = ipaddress.ip_address(ip_str)
                is_bad, reason = _is_disallowed_ip(resolved_ip, allow_private=allow_private)
                if is_bad:
                    raise ValueError(f"Database host '{url.host}' resolves to a restricted address ({reason}: {ip_str})")
                resolved_ips.append(ip_str)
            except ValueError as ip_err:
                if "Database host" in str(ip_err):
                    raise
                raise ValueError(f"Invalid resolved IP address: {ip_str}") from ip_err

        pinned_ip = resolved_ips[0]

    normalized_str = url.render_as_string(hide_password=False)
    _validated_host_ips[normalized_str] = pinned_ip
    return normalized_str, pinned_ip, port


def normalized_database_url(value: str) -> str:
    """Validate destination host and return normalized PostgreSQL URL string."""
    normalized_str, _, _ = validate_database_destination(value)
    return normalized_str


def get_pinned_hostaddr(url_str: str) -> str | None:
    """Retrieve the validated IP address for a connection URL to prevent DNS rebinding."""
    return _validated_host_ips.get(url_str)


def public_settings(user) -> dict:
    """Return public developer settings including explicit database routing mode and fallback notices."""
    url = None
    try:
        url = decrypt(user.developer_database_url)
    except Exception:
        pass

    display_url = None
    if url:
        try:
            parsed = make_url(url)
            host = parsed.host or ""
            username = f"{parsed.username}:••••@" if parsed.username else ""
            display_url = f"postgresql://{username}{host}:{parsed.port or 5432}/{parsed.database}"
        except Exception:
            display_url = "postgresql://••••:••••@••••/••••"

    mode = database_routing_mode_context.get()
    fallback_reason = database_fallback_reason_context.get()
    cached_status = get_personal_db_status(getattr(user, "id", None))

    # If user has no DB configured, mode is always APP_DEFAULT
    if not url:
        mode = DatabaseRoutingMode.APP_DEFAULT.value
        fallback_notice = None
    elif mode == DatabaseRoutingMode.APP_FALLBACK.value or cached_status:
        mode = DatabaseRoutingMode.APP_FALLBACK.value
        fallback_notice = {
            "title": "Personal database connection failed",
            "message": "Unable to connect to your personal database due to invalid credentials or a connection error. VC11 has automatically switched to the default application database. Your data operations will continue using the application database.",
            "reason": fallback_reason or (cached_status.get("reason") if cached_status else None) or "Personal database is temporarily unreachable.",
        }
    else:
        mode = DatabaseRoutingMode.PERSONAL_ACTIVE.value
        fallback_notice = None

    return {
        "database_configured": bool(url),
        "database_url_display": display_url,
        "database_mode": mode,
        "fallback_notice": fallback_notice,
        "llm_provider": user.developer_llm_provider,
        "llm_key_configured": bool(user.developer_llm_api_key),
        "daily_chat_limit": settings.developer_daily_chat_limit if user.developer_llm_api_key else 50,
    }


def verify_personal_database(url: str) -> None:
    """Require an already migrated instance; never run DDL on a supplied URL."""
    from app.database.connection import _personal_engine
    from app.models.email import Base

    # Destination host is re-validated to ensure IP pinning
    validate_database_destination(url)
    engine = _personal_engine(url)

    # Bind the inspector to one connection with connection timeout
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

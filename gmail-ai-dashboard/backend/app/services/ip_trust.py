"""Trusted client IP extraction, Cloudflare origin validation, and privacy-preserving IP hashing."""

import hashlib
import hmac
import ipaddress
import logging
import threading
import time
from typing import List

from fastapi import Request

from app.core.config import settings

logger = logging.getLogger(__name__)

# Cloudflare official IPv4 and IPv6 CIDR blocks
# (https://www.cloudflare.com/ips/)
CLOUDFLARE_IPV4_CIDRS = [
    "173.245.48.0/20",
    "103.21.244.0/22",
    "103.22.200.0/22",
    "103.31.4.0/22",
    "141.101.64.0/18",
    "108.162.192.0/18",
    "190.93.240.0/20",
    "188.114.96.0/20",
    "197.234.240.0/22",
    "198.41.128.0/17",
    "162.158.0.0/15",
    "104.16.0.0/13",
    "104.24.0.0/14",
    "172.64.0.0/13",
    "131.0.72.0/22",
]

CLOUDFLARE_IPV6_CIDRS = [
    "2400:cb00::/32",
    "2606:4700::/32",
    "2803:f800::/32",
    "2405:b500::/32",
    "2405:8100::/32",
    "2a06:98c0::/29",
    "2c0f:f248::/32",
]

_PARSED_CF_NETWORKS = [ipaddress.ip_network(cidr) for cidr in CLOUDFLARE_IPV4_CIDRS + CLOUDFLARE_IPV6_CIDRS]

# In-memory temporary IP block cache: {ip_hash: unblock_timestamp}
_TEMPORARY_IP_BLOCKS: dict[str, float] = {}
_IP_BLOCK_LOCK = threading.Lock()


def is_cloudflare_ip(ip_str: str) -> bool:
    """Check if the given IP belongs to official Cloudflare edge proxy ranges."""
    try:
        ip = ipaddress.ip_address(ip_str.strip())
        return any(ip in net for net in _PARSED_CF_NETWORKS)
    except ValueError:
        return False


def is_direct_origin(request: Request) -> bool:
    """Detect whether a request accessed the Heroku origin directly, bypassing Cloudflare.

    In production, legitimate users reach the application via https://mail-assistant.rohit-jadhav.dev,
    which is proxied by Cloudflare. A direct hit to the herokuapp.com domain or a missing CF-Ray
    header while targeting the production domain indicates a direct origin access attempt.
    """
    if settings.environment != "production":
        return False

    host = (request.headers.get("host") or "").split(":")[0].lower()
    if "herokuapp.com" in host:
        return True

    # Cloudflare proxy always injects CF-Ray and CF-Connecting-IP
    if not request.headers.get("cf-ray") or not request.headers.get("cf-connecting-ip"):
        return True

    # If an optional secret header is configured on Cloudflare Transform Rules, verify it
    if settings.cloudflare_proxy_secret:
        received_secret = request.headers.get("x-cf-secret", "")
        if not hmac.compare_digest(received_secret, settings.cloudflare_proxy_secret):
            return True

    return False


def extract_client_ip(request: Request) -> str:
    """Extract trusted client IP address.

    Security rules:
    1. Never blindly trust client-supplied X-Forwarded-For.
    2. Trust CF-Connecting-IP only if:
       - The immediate peer (request.client.host) is an official Cloudflare edge IP, OR
       - The peer is in settings.trusted_proxies, OR
       - Running in development / test environments.
    3. If untrusted proxy headers are provided without a trusted hop, fallback to immediate socket client.
    """
    client_host = request.client.host if request.client else "127.0.0.1"

    # Fast path for development / testing
    is_prod = settings.environment == "production"

    trusted_list: List[str] = []
    if settings.trusted_proxies:
        trusted_list = [p.strip() for p in settings.trusted_proxies.split(",") if p.strip()]

    # Check custom trusted list
    peer_is_trusted = client_host in trusted_list or is_cloudflare_ip(client_host)

    if is_prod:
        if peer_is_trusted:
            cf_ip = request.headers.get("cf-connecting-ip", "").strip()
            if cf_ip:
                try:
                    ipaddress.ip_address(cf_ip)
                    return cf_ip
                except ValueError:
                    pass

            xff = request.headers.get("x-forwarded-for", "")
            if xff:
                # Leftmost IP in XFF is the client IP added by first proxy
                candidate = xff.split(",")[0].strip()
                try:
                    ipaddress.ip_address(candidate)
                    return candidate
                except ValueError:
                    pass

        # If not from a trusted proxy, use direct client host (Heroku router)
        return client_host
    else:
        # Development / test environment
        cf_ip = request.headers.get("cf-connecting-ip", "").strip()
        if cf_ip:
            return cf_ip
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip()
        return client_host


def digest_ip(ip: str) -> str:
    """Hash IP address using keyed HMAC-SHA256 for privacy compliance.

    Raw IP addresses are never persisted indefinitely in long-term application logs.
    """
    secret = settings.session_secret
    return hmac.new(secret.encode(), f"vc11-ip:{ip}".encode(), hashlib.sha256).hexdigest()[:32]


def block_ip(ip_hash: str, duration_seconds: int = 900) -> None:
    """Temporarily block an IP hash for the specified duration (default 15 minutes)."""
    until = time.time() + duration_seconds
    with _IP_BLOCK_LOCK:
        _TEMPORARY_IP_BLOCKS[ip_hash] = until


def is_ip_blocked(ip_hash: str) -> tuple[bool, int]:
    """Check if an IP hash is in temporary cooldown. Returns (is_blocked, remaining_seconds)."""
    now = time.time()
    with _IP_BLOCK_LOCK:
        until = _TEMPORARY_IP_BLOCKS.get(ip_hash)
        if until is not None:
            if until > now:
                return True, int(until - now)
            del _TEMPORARY_IP_BLOCKS[ip_hash]
    return False, 0

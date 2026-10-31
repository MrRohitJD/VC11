"""Unified Multi-Layer Rate Limiting and Abuse Prevention Middleware for FastAPI."""

import logging
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.database.connection import SessionLocal
from app.services.ip_trust import (
    block_ip,
    digest_ip,
    extract_client_ip,
    is_direct_origin,
    is_ip_blocked,
)
from app.services.rate_limiter import check_rate_limit
from app.services.security_events import record_security_event
from app.services.visitor_service import (
    get_or_register_visitor,
    get_visitor_id,
    increment_abuse_score,
    set_visitor_cookie,
    validate_visitor_cookie,
    visitor_hash,
)

logger = logging.getLogger(__name__)

# Endpoints exempt from general rate limiting
EXEMPT_PREFIXES = (
    "/health",
    "/assets/",
    "/favicon.ico",
    "/robots.txt",
)

# Endpoint-specific rate limit configurations: (limit, window_seconds, key_prefix)
ENDPOINT_LIMITS = {
    # Auth endpoints
    ("POST", "/api/auth/demo"): (10, 60, "auth:demo"),
    ("GET", "/api/auth/google"): (15, 60, "auth:google"),
    ("GET", "/api/auth/callback"): (15, 60, "auth:callback"),
    ("GET", "/auth/google/callback"): (15, 60, "auth:callback"),
    ("POST", "/api/auth/logout"): (30, 60, "auth:logout"),
    ("DELETE", "/api/auth/erase"): (2, 3600, "auth:erase"),
    ("GET", "/api/auth/me"): (120, 60, "auth:me"),
    # Gmail sync & send
    ("POST", "/api/gmail/sync"): (getattr(settings, "gmail_sync_hourly_limit", 60), 3600, "gmail:sync"),
    ("POST", "/api/emails/send"): (15, 3600, "email:send"),
    ("POST", "/api/gmail/send"): (15, 3600, "email:send"),
    ("GET", "/api/emails"): (150, 60, "email:list"),
    ("GET", "/api/analytics/charts"): (60, 60, "email:charts"),
    ("GET", "/api/analytics/summary"): (60, 60, "email:summary"),
    # AI & RAG
    ("POST", "/api/rag/chat"): (15, 60, "rag:chat:burst"),
    ("POST", "/api/rag/chat/stream"): (15, 60, "rag:stream:burst"),
    ("POST", "/api/rag/index"): (getattr(settings, "rag_index_hourly_limit", 60), 3600, "rag:index"),
    # LLM classification (100 req/min per user)
    ("POST", "/api/llm/classify/backfill"): (getattr(settings, "llm_classification_per_minute", 100), 60, "llm:backfill"),
    ("POST", "/api/llm/classify/start"): (getattr(settings, "llm_classification_per_minute", 100), 60, "llm:start"),
    ("POST", "/api/llm/classify/all"): (getattr(settings, "llm_classification_per_minute", 100), 60, "llm:all"),
    ("POST", "/api/llm/train"): (getattr(settings, "llm_classification_per_minute", 100), 60, "llm:train"),
    ("POST", "/api/llm/classify/retry"): (getattr(settings, "llm_classification_per_minute", 100), 60, "llm:retry"),
    # Developer
    ("POST", "/api/developer/reconnect"): (10, 60, "dev:reconnect"),
    # Export
    ("GET", "/api/export/json"): (10, 3600, "export:json"),
    ("GET", "/api/export/csv"): (10, 3600, "export:csv"),
    ("POST", "/api/export/import"): (10, 3600, "export:import"),
}


def _match_endpoint_limit(method: str, path: str) -> Optional[Tuple[int, int, str]]:
    key = (method.upper(), path)
    rule = ENDPOINT_LIMITS.get(key)
    if not rule:
        # Check prefix matches for classification and export
        for (m, p), r in ENDPOINT_LIMITS.items():
            if m == method.upper() and path.startswith(p):
                rule = r
                break
    if not rule and method.upper() == "POST" and path.startswith("/api/llm/classify"):
        rule = (getattr(settings, "llm_classification_per_minute", 100), 60, "llm:classify")

    if rule:
        ep_limit, ep_win, ep_prefix = rule
        if ep_prefix == "gmail:sync":
            return (getattr(settings, "gmail_sync_hourly_limit", 60), ep_win, ep_prefix)
        if ep_prefix == "rag:index":
            return (getattr(settings, "rag_index_hourly_limit", 60), ep_win, ep_prefix)
        if ep_prefix.startswith("llm:"):
            return (getattr(settings, "llm_classification_per_minute", 100), ep_win, ep_prefix)
        return rule
    return None


class MultiLayerRateLimitMiddleware(BaseHTTPMiddleware):
    """Enforces multi-layer rate limits across Cloudflare IP, visitor, global, and endpoint quotas."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 1. Skip non-API routes and health checks
        if any(path.startswith(prefix) for prefix in EXEMPT_PREFIXES):
            return await call_next(request)

        # Bypass if globally disabled
        if not settings.rate_limit_enabled:
            return await call_next(request)

        client_ip = extract_client_ip(request)
        ip_h = digest_ip(client_ip)

        # 2. Check if IP is in temporary cooldown / block
        is_blocked, remaining_block = is_ip_blocked(ip_h)
        if is_blocked:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Access temporarily restricted due to excessive requests. Try again in {remaining_block} seconds.",
                    "retry_after": remaining_block,
                },
                headers={"Retry-After": str(remaining_block)},
            )

        # 3. Detect suspicious direct-to-origin bypass in production
        if is_direct_origin(request):
            logger.warning("Suspicious direct origin request detected from IP %s to %s", client_ip, path)

        # 4. Extract or validate visitor cookie
        raw_cookie = request.cookies.get(settings.demo_visitor_cookie_name, "")
        vid, is_tampered = validate_visitor_cookie(raw_cookie)
        v_h = visitor_hash(vid) if vid else None

        # Execute rate limit checks against PostgreSQL
        with SessionLocal() as db:
            if is_tampered:
                record_security_event(
                    db=db,
                    ip_hash=ip_h,
                    event_type="COOKIE_TAMPERING",
                    severity="WARNING",
                    endpoint=path,
                    http_method=request.method,
                    status_code=401,
                    detail="Tampered visitor cookie detected in request header",
                )

            # Layer 4: Global application rate limit
            g_allowed, g_retry, g_count, g_limit = check_rate_limit(
                db=db,
                bucket_key="global:all",
                limit=settings.rate_limit_global_per_minute,
                window_seconds=60,
                ip_hash=ip_h,
                visitor_hash=v_h,
                endpoint=path,
                http_method=request.method,
            )
            if not g_allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Application request capacity reached. Please retry shortly.",
                        "retry_after": g_retry,
                    },
                    headers={"Retry-After": str(g_retry)},
                )

            # Layer 2: IP Rate Limiting (Burst 1m and Sustained 1h)
            ip_b_allowed, ip_b_retry, ip_b_count, ip_b_limit = check_rate_limit(
                db=db,
                bucket_key=f"ip:burst:{ip_h}",
                limit=settings.rate_limit_ip_burst_per_minute,
                window_seconds=60,
                ip_hash=ip_h,
                visitor_hash=v_h,
                endpoint=path,
                http_method=request.method,
            )
            if not ip_b_allowed:
                # Progressive enforcement: trigger temporary block if burst is severely violated
                if ip_b_count > ip_b_limit * 2:
                    block_ip(ip_h, duration_seconds=300)
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Too many requests from this IP. Please wait {ip_b_retry} seconds.",
                        "retry_after": ip_b_retry,
                    },
                    headers={"Retry-After": str(ip_b_retry)},
                )

            ip_s_allowed, ip_s_retry, ip_s_count, ip_s_limit = check_rate_limit(
                db=db,
                bucket_key=f"ip:sustained:{ip_h}",
                limit=settings.rate_limit_ip_sustained_per_hour,
                window_seconds=3600,
                ip_hash=ip_h,
                visitor_hash=v_h,
                endpoint=path,
                http_method=request.method,
            )
            if not ip_s_allowed:
                block_ip(ip_h, duration_seconds=900)
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Hourly IP request limit exceeded. Cooldown active for {ip_s_retry} seconds.",
                        "retry_after": ip_s_retry,
                    },
                    headers={"Retry-After": str(ip_s_retry)},
                )

            # Layer 3: Visitor Rate Limiting (if visitor cookie present)
            if v_h:
                v_allowed, v_retry, v_count, v_limit = check_rate_limit(
                    db=db,
                    bucket_key=f"visitor:burst:{v_h}",
                    limit=settings.rate_limit_visitor_burst_per_minute,
                    window_seconds=60,
                    ip_hash=ip_h,
                    visitor_hash=v_h,
                    endpoint=path,
                    http_method=request.method,
                )
                if not v_allowed:
                    increment_abuse_score(db, v_h, ip_h, points=10, reason="Visitor burst limit exceeded", endpoint=path)
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": f"Visitor request burst limit exceeded. Try again in {v_retry} seconds.",
                            "retry_after": v_retry,
                        },
                        headers={"Retry-After": str(v_retry)},
                    )

            # Layer 5: Endpoint-Specific Rate Limiting
            rule = _match_endpoint_limit(request.method, path)
            endpoint_limit_header = None
            endpoint_remaining_header = None
            endpoint_reset_header = None

            if rule:
                ep_limit, ep_win, ep_prefix = rule
                # Use visitor hash for demo / anonymous or IP for auth/unauthenticated
                key_ident = v_h or ip_h
                ep_key = f"ep:{ep_prefix}:{key_ident}"
                ep_allowed, ep_retry, ep_count, ep_lim = check_rate_limit(
                    db=db,
                    bucket_key=ep_key,
                    limit=ep_limit,
                    window_seconds=ep_win,
                    ip_hash=ip_h,
                    visitor_hash=v_h,
                    endpoint=path,
                    http_method=request.method,
                )
                if not ep_allowed:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": f"Rate limit exceeded for this endpoint. Try again in {ep_retry} seconds.",
                            "retry_after": ep_retry,
                        },
                        headers={"Retry-After": str(ep_retry)},
                    )
                endpoint_limit_header = str(ep_lim)
                endpoint_remaining_header = str(max(0, ep_lim - ep_count))
                endpoint_reset_header = str(ep_win)

        # 5. Process request
        response = await call_next(request)

        # 6. Auto-issue visitor cookie for demo requests if missing
        if path.startswith("/api/auth/demo") or (path.startswith("/api/") and not vid and not is_tampered):
            set_visitor_cookie(response, vid)

        # 7. Add RFC standard rate limit headers
        if endpoint_limit_header:
            response.headers["X-RateLimit-Limit"] = endpoint_limit_header
            response.headers["X-RateLimit-Remaining"] = endpoint_remaining_header
            response.headers["X-RateLimit-Reset"] = endpoint_reset_header
        else:
            response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_ip_burst_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(max(0, settings.rate_limit_ip_burst_per_minute - ip_b_count))
            response.headers["X-RateLimit-Reset"] = "60"

        return response

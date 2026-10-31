"""AI token accounting, financial cost estimation, and daily/monthly budget enforcement."""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID, uuid4

from fastapi import HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.email import User
from app.models.rate_limit import AiUsageLog
from app.services.ip_trust import digest_ip, extract_client_ip
from app.services.security_events import record_security_event
from app.services.visitor_service import get_visitor_id, visitor_hash

logger = logging.getLogger(__name__)

# Model pricing in USD per 1,000,000 tokens
MODEL_PRICING = {
    # Azure OpenAI / OpenAI GPT-4.1-mini
    "gpt-4.1-mini": {"input_per_million": 0.15, "output_per_million": 0.60},
    "openai/gpt-4.1-mini": {"input_per_million": 0.15, "output_per_million": 0.60},
    "azure/gpt-4.1-mini": {"input_per_million": 0.15, "output_per_million": 0.60},
    # Embeddings
    "text-embedding-3-large": {"input_per_million": 0.13, "output_per_million": 0.0},
    "azure/text-embedding-3-large": {"input_per_million": 0.13, "output_per_million": 0.0},
    # Google Gemini 2.5 Flash
    "gemini-2.5-flash": {"input_per_million": 0.075, "output_per_million": 0.30},
    "google/gemini-2.5-flash": {"input_per_million": 0.075, "output_per_million": 0.30},
}

DEFAULT_INPUT_PER_M = 0.20
DEFAULT_OUTPUT_PER_M = 0.60


def estimate_tokens(text_content: str) -> int:
    """Rough conservative estimation of token count (~3.5 to 4 characters per token)."""
    if not text_content:
        return 0
    return max(1, math.ceil(len(text_content) / 3.5))


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate estimated cost in USD based on provider pricing tables."""
    pricing = MODEL_PRICING.get(model_name.lower())
    if pricing is None:
        # Check partial match
        for k, v in MODEL_PRICING.items():
            if k in model_name.lower():
                pricing = v
                break

    if pricing:
        input_rate = pricing.get("input_per_million", DEFAULT_INPUT_PER_M)
        output_rate = pricing.get("output_per_million", DEFAULT_OUTPUT_PER_M)
    else:
        input_rate = DEFAULT_INPUT_PER_M
        output_rate = DEFAULT_OUTPUT_PER_M

    input_cost = (prompt_tokens / 1_000_000.0) * input_rate
    output_cost = (completion_tokens / 1_000_000.0) * output_rate
    return round(input_cost + output_cost, 6)


def check_ai_budget(
    db: Session,
    user: User,
    request: Request,
    query_text: str,
) -> None:
    """Evaluate pre-call token and financial budgets before dispatching LLM queries.

    Raises:
        HTTPException(400) if query exceeds maximum allowed characters.
        HTTPException(429) if daily token limit or global USD budget is exhausted.
    """
    # 1. Guard against oversized prompt attacks
    if len(query_text) > settings.max_query_characters:
        client_ip = extract_client_ip(request)
        ip_h = digest_ip(client_ip)
        vid, _ = get_visitor_id(request)
        v_h = visitor_hash(vid) if vid else None
        req_path = request.url.path if hasattr(request, "scope") and "path" in request.scope else "/api/rag/chat"
        req_method = request.scope.get("method", "POST") if hasattr(request, "scope") and isinstance(request.scope, dict) else "POST"

        record_security_event(
            db=db,
            ip_hash=ip_h,
            visitor_hash=v_h,
            user_id=user.id if not user.is_demo else None,
            event_type="PROMPT_OVERSIZED",
            severity="WARNING",
            endpoint=req_path,
            http_method=req_method,
            status_code=400,
            detail=f"Query rejected: length {len(query_text)} chars exceeds max {settings.max_query_characters}",
        )
        raise HTTPException(
            status_code=400,
            detail=f"Query exceeds the maximum allowed length of {settings.max_query_characters} characters.",
        )

    now = datetime.now(timezone.utc)
    day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)

    # 2. Check visitor or user daily token budget
    estimated_req_tokens = estimate_tokens(query_text)

    if user.is_demo:
        vid, _ = get_visitor_id(request)
        v_h = visitor_hash(vid) if vid else "anonymous"
        daily_used_tokens = db.scalar(
            select(func.sum(AiUsageLog.total_tokens)).where(
                AiUsageLog.visitor_hash == v_h,
                AiUsageLog.created_at >= day_start,
            )
        ) or 0

        if daily_used_tokens + estimated_req_tokens > settings.visitor_daily_token_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Daily AI token limit reached ({settings.visitor_daily_token_limit} tokens per visitor). Try again tomorrow or sign in.",
                headers={"Retry-After": "3600"},
            )
    else:
        daily_used_tokens = db.scalar(
            select(func.sum(AiUsageLog.total_tokens)).where(
                AiUsageLog.user_id == user.id,
                AiUsageLog.created_at >= day_start,
            )
        ) or 0

        if daily_used_tokens + estimated_req_tokens > settings.user_daily_token_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Daily AI token limit reached ({settings.user_daily_token_limit} tokens per account). Limit resets at midnight UTC.",
                headers={"Retry-After": "3600"},
            )

    # 3. Check global daily and monthly financial budgets (all traffic)
    check_global_ai_budget(db)


def check_global_ai_budget(db: Session) -> None:
    """Evaluate global daily and monthly financial spend caps across all AI operations."""
    now = datetime.now(timezone.utc)
    day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)

    daily_cost = db.scalar(
        select(func.sum(AiUsageLog.estimated_cost_usd)).where(
            AiUsageLog.created_at >= day_start,
        )
    ) or 0.0

    if daily_cost >= settings.ai_daily_budget_usd:
        raise HTTPException(
            status_code=429,
            detail="The application's global daily AI budget has been reached. Please try again after midnight UTC.",
            headers={"Retry-After": "3600"},
        )

    monthly_cost = db.scalar(
        select(func.sum(AiUsageLog.estimated_cost_usd)).where(
            AiUsageLog.created_at >= month_start,
        )
    ) or 0.0

    if monthly_cost >= settings.ai_monthly_budget_usd:
        raise HTTPException(
            status_code=429,
            detail="The application's global monthly AI budget has been reached. Please contact administrator.",
            headers={"Retry-After": "86400"},
        )


def log_ai_usage(
    db: Session,
    operation_type: str,
    model_name: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
    ip_hash: str,
    user_id: Optional[UUID] = None,
    visitor_hash_val: Optional[str] = None,
) -> Optional[AiUsageLog]:
    """Persist actual token accounting and estimated USD cost to database."""
    total = prompt_tokens + completion_tokens
    cost = calculate_cost(model_name, prompt_tokens, completion_tokens)
    now = datetime.now(timezone.utc)

    try:
        log_entry = AiUsageLog(
            id=uuid4(),
            user_id=user_id,
            visitor_hash=visitor_hash_val,
            ip_hash=ip_hash,
            operation_type=operation_type,
            model_name=model_name,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            estimated_cost_usd=cost,
            created_at=now,
        )
        db.add(log_entry)
        db.commit()
        logger.info(
            "AI usage logged: op=%s model=%s tokens=%d cost=$%.6f",
            operation_type,
            model_name,
            total,
            cost,
        )
        return log_entry
    except Exception as exc:
        db.rollback()
        logger.warning("Could not log AI usage: %s", type(exc).__name__)
        return None

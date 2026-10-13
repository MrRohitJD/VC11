"""LLM classification API routes.

Endpoints:
  GET  /api/llm/preference            – read AI categorization preference
  PUT  /api/llm/preference            – update preference
  POST /api/llm/classify/backfill     – start backfill for existing unclassified emails
  POST /api/llm/classify/retry        – retry last failed/partial job
  GET  /api/llm/classify/status       – current job state (in-memory + DB)
  GET  /api/llm/classify/jobs         – job history
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.connection import get_db
from app.models.email import GmailAccount, User
from app.models.llm import LlmClassificationJob
from app.services.llm_classification_service import (
    enqueue_classification_job,
    get_job_state,
    get_latest_job,
    retry_failed_job,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["llm"])


def _get_account(db: Session, user: User) -> GmailAccount:
    account = db.scalar(select(GmailAccount).where(GmailAccount.user_id == user.id))
    if account is None:
        raise HTTPException(status_code=404, detail="No Gmail account linked")
    return account


# ---------------------------------------------------------------------------
# Preference endpoints
# ---------------------------------------------------------------------------

@router.get("/preference")
def get_preference(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the AI categorization preference for the user's Gmail account."""
    account = _get_account(db, user)
    return {
        "ai_classification_enabled": account.ai_classification_enabled,
        "gmail_account_email": account.email,
    }


@router.put("/preference")
def update_preference(
    enabled: bool,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update the AI categorization preference. Does not start or stop running jobs."""
    account = _get_account(db, user)
    account.ai_classification_enabled = enabled
    db.commit()
    logger.info("User %s set ai_classification_enabled=%s for account %s", user.id, enabled, account.id)
    return {
        "ai_classification_enabled": account.ai_classification_enabled,
        "message": "Preference saved",
    }


# ---------------------------------------------------------------------------
# Classification & Training endpoints
# ---------------------------------------------------------------------------

@router.post("/classify/backfill", status_code=status.HTTP_202_ACCEPTED)
@router.post("/classify/start", status_code=status.HTTP_202_ACCEPTED)
@router.post("/classify/all", status_code=status.HTTP_202_ACCEPTED)
@router.post("/train", status_code=status.HTTP_202_ACCEPTED)
def start_backfill(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Schedule background classification for all unclassified emails for the user."""
    account = _get_account(db, user)

    # Auto-enable classification preference
    if not account.ai_classification_enabled:
        account.ai_classification_enabled = True
        db.commit()

    # Trigger global daemon and enqueue account job
    from app.services.llm_classification_service import trigger_immediate_classification
    trigger_immediate_classification()

    job = enqueue_classification_job(account.id, job_type="MANUAL_TRAIN")
    if job is None:
        return {
            "message": "All emails already classified with AI labels.",
            "job_id": None,
        }

    return {
        "message": "AI classification started in background for all emails.",
        "job_id": str(job.id),
        "emails_discovered": job.emails_discovered,
    }


# ---------------------------------------------------------------------------
# Retry endpoint
# ---------------------------------------------------------------------------

@router.post("/classify/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_classification(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retry the last failed or partial classification job."""
    account = _get_account(db, user)

    if not account.ai_classification_enabled:
        raise HTTPException(
            status_code=403,
            detail="AI categorization is disabled. Enable it before retrying.",
        )

    job = retry_failed_job(db, account.id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="No failed or partial job to retry.",
        )

    return {
        "message": "Retry scheduled",
        "job_id": str(job.id),
    }


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------

@router.get("/classify/status")
def classification_status(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the current classification state (in-memory) and the latest DB job."""
    account = _get_account(db, user)
    in_memory = get_job_state(str(account.id))
    latest = get_latest_job(db, account.id)
    return {
        "ai_classification_enabled": account.ai_classification_enabled,
        "live": in_memory,
        "latest_job": _serialize_job(latest) if latest else None,
    }


# ---------------------------------------------------------------------------
# Job history
# ---------------------------------------------------------------------------

@router.get("/classify/jobs")
def classification_jobs(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the last 20 classification jobs for the user's Gmail account."""
    account = _get_account(db, user)
    jobs = db.scalars(
        select(LlmClassificationJob)
        .where(LlmClassificationJob.gmail_account_id == account.id)
        .order_by(LlmClassificationJob.created_at.desc())
        .limit(20)
    ).all()
    return {"jobs": [_serialize_job(j) for j in jobs]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_job(job: LlmClassificationJob) -> dict:
    return {
        "id": str(job.id),
        "job_type": job.job_type,
        "status": job.status,
        "emails_discovered": job.emails_discovered,
        "emails_classified": job.emails_classified,
        "emails_remaining": job.emails_remaining,
        "emails_failed": job.emails_failed,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "last_error": job.last_error,
    }

"""LLM email classification service.

Handles all three classification scenarios:
  A. Initial sync  – called from gmail sync callback
  B. Backfill      – classify historical emails that pre-date the feature
  C. Incremental   – classify newly fetched emails during routine sync

Jobs are persisted in ``llm_classification_jobs`` so they survive restarts.
Each batch is committed independently; no DB transaction is held during LLM calls.
"""

from __future__ import annotations

import logging
from contextvars import copy_context
import threading
import time
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import exists, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.connection import SessionLocal, CentralSessionLocal
from app.models.email import Email, EmailLabel, GmailAccount, Label, User, utcnow
from app.models.llm import LlmClassificationJob, LlmEmailClassification, LlmLabel
from app.services.llm_provider import PROMPT_VERSION, classify_emails_batch
from app.services.developer_settings import decrypt, database_url_context, llm_key_context

logger = logging.getLogger(__name__)

# Default batch size for LLM requests
_BATCH_SIZE = 15

# In-memory job state per account (keyed by str(gmail_account_id))
_job_state: dict[str, dict] = {}
_state_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Category seed
# ---------------------------------------------------------------------------

CATEGORY_SEEDS: list[tuple[str, str]] = [
    ("WORK", "Professional work tasks and internal company communications."),
    ("JOB", "Job applications, interview invites, recruiter outreach."),
    ("EDUCATION", "Course content, university, school, learning platforms."),
    ("FINANCE", "Bank statements, investment reports, financial news."),
    ("BILLS", "Utility bills, invoices, payment reminders."),
    ("PROMOTIONS", "Marketing emails, sale announcements, discount codes."),
    ("SOCIAL", "Social networks, friend requests, community notifications."),
    ("TRAVEL", "Flight confirmations, hotel bookings, itineraries."),
    ("PERSONAL", "Personal correspondence from friends and family."),
    ("UPDATES", "App update notifications, platform announcements."),
    ("SPAM", "Unsolicited bulk email, phishing attempts."),
    ("OTHER", "Does not clearly fit any other category."),
    ("MEETINGS", "Meeting invitations, video calls, Zoom/Teams links."),
    ("CALENDAR", "Calendar events, reminders, scheduling."),
    ("PROJECTS", "Project management notifications, Jira/Asana/GitHub."),
    ("TECHNICAL", "Technical alerts, server logs, dev tools, monitoring."),
    ("SECURITY", "Security alerts, breach notices, 2FA, password resets."),
    ("ACCOUNT", "Account creation, verification, profile updates."),
    ("ORDERS", "Purchase confirmations, order updates, e-commerce."),
    ("SHIPPING", "Shipping notifications, delivery tracking."),
    ("SUBSCRIPTIONS", "Subscription renewals, plan changes, trial expirations."),
    ("CUSTOMER_SUPPORT", "Support tickets, help desk responses."),
    ("HEALTH", "Healthcare appointments, medical records, pharmacy."),
    ("LEGAL", "Contracts, legal notices, compliance."),
    ("GOVERNMENT", "Government agencies, tax authorities, civic notices."),
    ("NEWSLETTERS", "Newsletters, digests, editorial content."),
    ("EVENTS", "Event invitations, conferences, webinars, meetups."),
    ("RECEIPTS", "Payment receipts, expense records."),
    ("RECRUITMENT_UPDATES", "Recruiter status updates, application progress."),
    ("AI_DEVELOPMENT", "AI/ML tools, model updates, AI product emails."),
]


def seed_llm_labels(db: Session) -> dict[str, LlmLabel]:
    """Idempotently insert all 30 approved categories and return a name→model map."""
    label_map: dict[str, LlmLabel] = {}
    for name, description in CATEGORY_SEEDS:
        existing = db.scalar(select(LlmLabel).where(LlmLabel.name == name))
        if existing is None:
            existing = LlmLabel(name=name, description=description)
            db.add(existing)
            db.flush()
        label_map[name] = existing
    db.commit()
    return label_map


def get_label_map(db: Session) -> dict[str, LlmLabel]:
    """Return a name→LlmLabel map for all seeded categories."""
    rows = db.scalars(select(LlmLabel)).all()
    return {row.name: row for row in rows}


# ---------------------------------------------------------------------------
# Query: find unclassified emails for a Gmail account
# ---------------------------------------------------------------------------

def _fetch_unclassified_batch(
    db: Session,
    gmail_account_id: UUID,
    batch_size: int = _BATCH_SIZE,
    offset: int = 0,
) -> list[tuple[UUID, str | None]]:
    """Return email metadata tuples for emails that have no LLM classification.

    Uses NOT EXISTS anti-join so the same email is never processed twice.
    Scoped to the specified Gmail account to prevent cross-account access.
    """
    stmt = (
        select(
            Email.id,
            Email.subject,
            Email.body,
            Email.snippet,
            Email.from_email,
        )
        .where(
            Email.gmail_account_id == gmail_account_id,
            ~exists().where(
                LlmEmailClassification.email_id == Email.id,
                LlmEmailClassification.is_current.is_(True),
                LlmEmailClassification.classification_status == "SUCCEEDED",
            ),
        )
        .order_by(Email.created_at.asc(), Email.id.asc())
        .limit(batch_size)
        .offset(offset)
    )
    return list(db.execute(stmt).all())


def _count_unclassified(db: Session | None = None, gmail_account_id: UUID | None = None) -> int:
    """Count emails that have no current SUCCEEDED LLM classification."""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    try:
        stmt = (
            select(Email.id)
            .where(
                ~exists().where(
                    LlmEmailClassification.email_id == Email.id,
                    LlmEmailClassification.is_current.is_(True),
                    LlmEmailClassification.classification_status == "SUCCEEDED",
                ),
            )
        )
        if gmail_account_id is not None:
            stmt = stmt.where(Email.gmail_account_id == gmail_account_id)
        count_stmt = select(func.count()).select_from(stmt.subquery())
        return db.scalar(count_stmt) or 0
    finally:
        if close_db:
            db.close()


# ---------------------------------------------------------------------------
# Persist a single batch of classification results
# ---------------------------------------------------------------------------

def _persist_batch(
    db: Session,
    label_map: dict[str, LlmLabel],
    classifications: list,
    model_name: str,
    valid_email_ids: set[str],
) -> tuple[int, int]:
    """Upsert classification results into email_classifications, labels, and email_labels."""
    saved = 0
    failed = 0
    for item in classifications:
        email_id_str = item.email_id
        if email_id_str not in valid_email_ids:
            logger.warning("LLM returned unknown email_id %s — skipping", email_id_str)
            failed += 1
            continue

        label_name = item.label if item.label in label_map else "OTHER"
        label_obj = label_map.get(label_name)
        if label_obj is None and label_map:
            label_obj = next(iter(label_map.values()))
        if label_obj is None:
            logger.error("LLM label %r not in DB — skipping email %s", label_name, email_id_str)
            failed += 1
            continue

        try:
            email_uuid = UUID(email_id_str)
        except ValueError:
            failed += 1
            continue

        try:
            existing = db.scalar(
                select(LlmEmailClassification).where(
                    LlmEmailClassification.email_id == email_uuid,
                    LlmEmailClassification.is_current.is_(True),
                )
            )
            now = utcnow()
            if existing is None:
                row = LlmEmailClassification(
                    email_id=email_uuid,
                    label_id=label_obj.id,
                    model_name=model_name,
                    prompt_version=PROMPT_VERSION,
                    confidence=item.confidence,
                    reason=item.reason,
                    classification_status="SUCCEEDED",
                    is_current=True,
                    classified_at=now,
                    created_at=now,
                    updated_at=now,
                )
                db.add(row)
            else:
                existing.label_id = label_obj.id
                existing.model_name = model_name
                existing.prompt_version = PROMPT_VERSION
                existing.confidence = item.confidence
                existing.reason = item.reason
                existing.classification_status = "SUCCEEDED"
                existing.classified_at = now
                existing.updated_at = now

            # Ensure the AI category label is also in labels and email_labels tables
            email_obj = db.get(Email, email_uuid)
            if email_obj:
                account_label = db.scalar(
                    select(Label).where(
                        Label.gmail_account_id == email_obj.gmail_account_id,
                        Label.name == label_name,
                    )
                )
                if account_label is None:
                    account_label = Label(
                        gmail_account_id=email_obj.gmail_account_id,
                        label_id=f"AI_{label_name}",
                        name=label_name,
                        type="ai",
                    )
                    db.add(account_label)
                    db.flush()

                email_label = db.scalar(
                    select(EmailLabel).where(
                        EmailLabel.email_id == email_uuid,
                        EmailLabel.label_id == account_label.id,
                    )
                )
                if email_label is None:
                    db.add(
                        EmailLabel(
                            email_id=email_uuid,
                            label_id=account_label.id,
                            gmail_account_id=email_obj.gmail_account_id,
                        )
                    )

            db.flush()
            saved += 1
        except Exception as exc:
            logger.warning("Failed to persist classification for %s: %s", email_id_str, exc)
            db.rollback()
            failed += 1

    try:
        db.commit()
    except Exception as exc:
        logger.error("Batch commit failed: %s", exc)
        db.rollback()
        return 0, len(classifications)

    return saved, failed


# ---------------------------------------------------------------------------
# Core classification worker
# ---------------------------------------------------------------------------

def _run_classification_job(job_id: UUID, gmail_account_id: UUID) -> None:
    """Run the classification job in a background thread.

    Fetches unclassified emails in batches, calls the LLM, persists results,
    and updates the job record throughout. Safe to resume after restart —
    already-classified emails are skipped by the NOT EXISTS query.
    """
    db = SessionLocal()
    key_token = None
    account_id_str = str(gmail_account_id)
    try:
        account = db.get(GmailAccount, gmail_account_id)
        if account is not None and llm_key_context.get() is None:
            with CentralSessionLocal() as central:
                owner = central.get(User, account.user_id)
                if owner is not None and owner.developer_llm_api_key:
                    key_token = llm_key_context.set((
                        owner.developer_llm_provider, decrypt(owner.developer_llm_api_key)))
        # Mark job as RUNNING
        job = db.get(LlmClassificationJob, job_id)
        if job is None:
            logger.error("Classification job %s not found", job_id)
            return

        # Claim atomically. A second worker sees no PENDING row and exits.
        claimed = db.execute(
            update(LlmClassificationJob)
            .where(LlmClassificationJob.id == job_id, LlmClassificationJob.status == "PENDING")
            .values(status="RUNNING", started_at=utcnow(), attempts=LlmClassificationJob.attempts + 1)
        )
        if claimed.rowcount != 1:
            logger.info("Classification job %s was already claimed", job_id)
            return
        db.commit()
        db.refresh(job)

        # Refresh label map
        label_map = get_label_map(db)
        if not label_map:
            label_map = seed_llm_labels(db)

        total_unclassified = _count_unclassified(db, gmail_account_id)
        job.emails_discovered = total_unclassified
        job.emails_remaining = total_unclassified
        db.commit()

        _update_job_state(account_id_str, {
            "status": "running",
            "emails_discovered": total_unclassified,
            "emails_classified": 0,
            "emails_remaining": total_unclassified,
            "emails_failed": 0,
        })

        total_classified = 0
        total_failed = 0
        disabled_providers: set[str] = set()

        while True:
            # Always fetch from offset=0 — already-classified are excluded by NOT EXISTS
            batch = _fetch_unclassified_batch(db, gmail_account_id, batch_size=_BATCH_SIZE)
            if not batch:
                break

            email_payloads = [
                {
                    "email_id": str(row[0]),
                    "subject": row[1] or "",
                    "body": row[2] or "",
                    "snippet": row[3] or "",
                    "from_email": row[4] or "",
                }
                for row in batch
            ]
            valid_ids = {p["email_id"] for p in email_payloads}
            category_descriptions = {
                name: label.description or "" for name, label in label_map.items()
            }

            try:
                # LLM call OUTSIDE any open transaction
                results, model_name = classify_emails_batch(
                    email_payloads,
                    category_descriptions,
                    disabled_providers=disabled_providers,
                )
            except Exception as exc:
                logger.error("LLM batch failed for job %s: %s — falling back to local classification for batch", job_id, exc)
                from app.services.llm_provider import _local_classify
                results, model_name = _local_classify(email_payloads, category_descriptions)

            batch_saved, batch_failed = _persist_batch(db, label_map, results, model_name, valid_ids)
            total_classified += batch_saved
            total_failed += batch_failed

            remaining = _count_unclassified(db, gmail_account_id)
            job.emails_classified = total_classified
            job.emails_remaining = remaining
            job.emails_failed = total_failed
            db.commit()

            _update_job_state(account_id_str, {
                "status": "running",
                "emails_discovered": total_unclassified,
                "emails_classified": total_classified,
                "emails_remaining": remaining,
                "emails_failed": total_failed,
            })

            if not batch:
                break

        # Finalize job
        final_remaining = _count_unclassified(db, gmail_account_id)
        job.emails_classified = total_classified
        job.emails_remaining = final_remaining
        job.emails_failed = total_failed
        job.completed_at = datetime.now(timezone.utc)

        if total_failed > 0 and total_classified == 0:
            job.status = "FAILED"
        elif total_failed > 0:
            job.status = "PARTIAL"
        else:
            job.status = "COMPLETED"

        db.commit()

        _update_job_state(account_id_str, {
            "status": job.status.lower(),
            "emails_discovered": total_unclassified,
            "emails_classified": total_classified,
            "emails_remaining": final_remaining,
            "emails_failed": total_failed,
            "completed_at": job.completed_at.isoformat(),
        })
        logger.info(
            "Classification job %s %s: classified=%d failed=%d remaining=%d",
            job_id, job.status, total_classified, total_failed, final_remaining,
        )

    except Exception as exc:
        logger.exception("Classification job %s crashed: %s", job_id, exc)
        try:
            job = db.get(LlmClassificationJob, job_id)
            if job:
                job.status = "FAILED"
                job.last_error = str(exc)[:500]
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass
        _update_job_state(account_id_str, {"status": "failed", "error": str(exc)[:300]})
    finally:
        if key_token is not None:
            llm_key_context.reset(key_token)
        db.close()


def _update_job_state(account_id_str: str, update_dict: dict) -> None:
    with _state_lock:
        state = _job_state.setdefault(account_id_str, {})
        state.update(update_dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_job_state(gmail_account_id: str) -> dict:
    """Return current in-memory classification state for an account."""
    with _state_lock:
        return dict(_job_state.get(gmail_account_id, {
            "status": "idle",
            "emails_discovered": 0,
            "emails_classified": 0,
            "emails_remaining": 0,
            "emails_failed": 0,
        }))


def enqueue_classification_job(
    gmail_account_id: UUID,
    job_type: str = "INCREMENTAL",
) -> LlmClassificationJob | None:
    """Create a durable classification job and start it in a background thread.

    Returns the job or None if no unclassified emails exist.
    Thread safety: a new thread is started; a single in-progress job per account
    is enforced by checking existing RUNNING jobs.
    """
    db = SessionLocal()
    try:
        account = db.get(GmailAccount, gmail_account_id)
        if account is None:
            logger.error("enqueue_classification_job: account %s not found", gmail_account_id)
            return None

        # Don't start a second job if one is already running
        running = db.scalar(
            select(LlmClassificationJob).where(
                LlmClassificationJob.gmail_account_id == gmail_account_id,
                LlmClassificationJob.status == "RUNNING",
            )
        )
        if running:
            live_status = get_job_state(str(gmail_account_id)).get("status")
            if live_status in {"queued", "running"}:
                logger.info("Classification already running for account %s — skipping", gmail_account_id)
                return running
            # A process restart can leave a durable job RUNNING after its
            # daemon worker disappeared. Requeue it so the next sync resumes.
            running.status = "PENDING"
            running.started_at = None
            running.last_error = "Recovered after an interrupted worker."
            db.commit()

        # Check if there's anything to classify
        unclassified_count = _count_unclassified(db, gmail_account_id)
        if unclassified_count == 0:
            logger.info("No unclassified emails for account %s", gmail_account_id)
            return None

        job = LlmClassificationJob(
            gmail_account_id=gmail_account_id,
            job_type=job_type,
            status="PENDING",
            emails_discovered=unclassified_count,
            emails_remaining=unclassified_count,
        )
        db.add(job)
        try:
            db.commit()
        except IntegrityError:
            # The partial unique index is the final guard when two requests race.
            db.rollback()
            return db.scalar(
                select(LlmClassificationJob).where(
                    LlmClassificationJob.gmail_account_id == gmail_account_id,
                    LlmClassificationJob.job_type == job_type,
                    LlmClassificationJob.email_id.is_(None),
                    LlmClassificationJob.status.in_(("PENDING", "RUNNING")),
                )
            )
        db.refresh(job)
        job_id = job.id
    finally:
        db.close()

    account_id_str = str(gmail_account_id)
    _update_job_state(account_id_str, {"status": "queued", "job_id": str(job_id)})

    context = copy_context()
    thread = threading.Thread(
        target=lambda: context.run(_run_classification_job, job_id, gmail_account_id),
        name=f"llm-classify-{job_type}-{str(job_id)[:8]}",
        daemon=True,
    )
    thread.start()
    return job


def get_latest_job(db: Session, gmail_account_id: UUID) -> LlmClassificationJob | None:
    """Return the most recent job for an account."""
    return db.scalar(
        select(LlmClassificationJob)
        .where(LlmClassificationJob.gmail_account_id == gmail_account_id)
        .order_by(LlmClassificationJob.created_at.desc())
        .limit(1)
    )


def retry_failed_job(db: Session, gmail_account_id: UUID) -> LlmClassificationJob | None:
    """Re-enqueue if the latest job is FAILED or PARTIAL."""
    job = get_latest_job(db, gmail_account_id)
    if job is None or job.status not in ("FAILED", "PARTIAL"):
        return None
    return enqueue_classification_job(gmail_account_id, job_type=job.job_type)


# ---------------------------------------------------------------------------
# Global Background Daemon & Event-Driven Triggers
# ---------------------------------------------------------------------------

_daemon_started = False
_daemon_wake_event = threading.Event()


def trigger_immediate_classification() -> None:
    """Signal the background classification daemon to wake up immediately."""
    _daemon_wake_event.set()


def ensure_user_emails_classified(gmail_account_id: UUID) -> None:
    """Non-blocking check for logged-in user: triggers classification if unclassified emails exist."""
    count = _count_unclassified(None, gmail_account_id)
    if count > 0:
        enqueue_classification_job(gmail_account_id, job_type="USER_SYNC")
        trigger_immediate_classification()


def _run_global_classification_cycle() -> None:
    """Check for any unclassified emails across all accounts and enqueue workers."""
    db = SessionLocal()
    try:
        unclassified_count = _count_unclassified(db)
        if unclassified_count:
            logger.info(
                "Background AI daemon detected %d unclassified emails across accounts. Starting classification...",
                unclassified_count,
            )

        account_ids = db.scalars(
            select(Email.gmail_account_id)
            .where(
                ~exists().where(
                    LlmEmailClassification.email_id == Email.id,
                    LlmEmailClassification.is_current.is_(True),
                    LlmEmailClassification.classification_status == "SUCCEEDED",
                )
            )
            .distinct()
        ).all()

        for account_id in account_ids:
            # Auto-enable ai_classification_enabled for all accounts
            account = db.get(GmailAccount, account_id)
            if (account and database_url_context.get() is None
                    and db.get(User, account.user_id).developer_database_url):
                continue
            if account and not account.ai_classification_enabled:
                account.ai_classification_enabled = True
                db.commit()

            enqueue_classification_job(account_id, job_type="DAEMON_SWEEP")
    finally:
        db.close()
    if database_url_context.get() is None:
        with CentralSessionLocal() as central:
            personal_urls = [decrypt(value) for value in central.scalars(
                select(User.developer_database_url).where(
                    User.developer_database_url.is_not(None))).all()]
        for url in personal_urls:
            token = database_url_context.set(url)
            try:
                _run_global_classification_cycle()
            except Exception as exc:
                logger.warning("Personal classification sweep failed: %s", type(exc).__name__)
            finally:
                database_url_context.reset(token)


def _global_ai_classifier_loop() -> None:
    """Continuous daemon loop that sweeps for unclassified emails every 15s."""
    logger.info("Global background AI classifier daemon loop started.")
    time.sleep(2)
    while True:
        try:
            _daemon_wake_event.clear()
            _run_global_classification_cycle()
        except Exception as exc:
            logger.exception("Error in background AI classification daemon: %s", exc)

        # Check every 15 seconds, or wake up immediately if triggered by a sync / import
        _daemon_wake_event.wait(timeout=15.0)


def start_global_ai_classifier_daemon() -> None:
    """Idempotently start the continuous background daemon thread."""
    global _daemon_started
    with _state_lock:
        if _daemon_started:
            return
        _daemon_started = True

    thread = threading.Thread(
        target=_global_ai_classifier_loop,
        name="global-ai-classifier-daemon",
        daemon=True,
    )
    thread.start()
    logger.info("Started global background AI classifier daemon.")

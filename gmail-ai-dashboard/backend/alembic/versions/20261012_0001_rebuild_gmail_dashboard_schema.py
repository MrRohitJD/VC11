"""Create the normalized Gmail AI Dashboard schema.

This revision is intentionally additive: it creates tables in the existing
database and never drops/recreates the database itself. It is for an empty
database; an installation with legacy tables must be reconciled deliberately
before applying this initial revision.
"""

from __future__ import annotations

from uuid import uuid4

from alembic import op

from app.models.email import AiCategory, Base
import app.models.llm  # noqa: F401 - register llm_classification_jobs


revision = "20261012_0001"
down_revision = None
branch_labels = None
depends_on = None


CATEGORY_SEEDS = [
    ("WORK", "Professional work tasks and internal company communications."),
    ("JOB", "Job applications, interview invites, and recruiter outreach."),
    ("EDUCATION", "Courses, university, school, and learning platforms."),
    ("FINANCE", "Bank statements, investment reports, and financial news."),
    ("BILLS", "Utility bills, invoices, and payment reminders."),
    ("PROMOTIONS", "Marketing, sales, and discount communications."),
    ("SOCIAL", "Social networks, friends, and community notifications."),
    ("TRAVEL", "Flights, hotels, itineraries, and travel updates."),
    ("PERSONAL", "Personal correspondence from friends and family."),
    ("UPDATES", "Application and platform announcements."),
    ("SPAM", "Unsolicited bulk email and phishing attempts."),
    ("OTHER", "Does not clearly fit another approved category."),
    ("MEETINGS", "Meeting invitations and video-call links."),
    ("CALENDAR", "Calendar events, reminders, and scheduling."),
    ("PROJECTS", "Project-management notifications."),
    ("TECHNICAL", "Technical alerts, logs, and developer tooling."),
    ("SECURITY", "Security alerts, breach notices, and password resets."),
    ("ACCOUNT", "Account creation, verification, and profile updates."),
    ("ORDERS", "Purchase confirmations and e-commerce updates."),
    ("SHIPPING", "Delivery and shipment tracking."),
    ("SUBSCRIPTIONS", "Renewals, plan changes, and trials."),
    ("CUSTOMER_SUPPORT", "Support tickets and help-desk responses."),
    ("HEALTH", "Healthcare appointments and pharmacy messages."),
    ("LEGAL", "Contracts, legal notices, and compliance."),
    ("GOVERNMENT", "Government agencies, tax, and civic notices."),
    ("NEWSLETTERS", "Newsletters, digests, and editorial content."),
    ("EVENTS", "Invitations, conferences, webinars, and meetups."),
    ("RECEIPTS", "Receipts and expense records."),
    ("RECRUITMENT_UPDATES", "Recruiter status and application progress."),
    ("AI_DEVELOPMENT", "AI/ML tools, model updates, and product messages."),
]


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind, checkfirst=True)

    # PostgreSQL and SQLite both support this partial unique-index syntax.
    # It permits full classification history while exactly one result is current.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_email_classifications_current "
        "ON email_classifications (email_id) WHERE is_current = true"
    )
    # One active batch of a given type per account and one active per-email job.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_llm_jobs_active_batch "
        "ON llm_classification_jobs (gmail_account_id, job_type) "
        "WHERE email_id IS NULL AND status IN ('PENDING', 'RUNNING')"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_llm_jobs_active_email "
        "ON llm_classification_jobs (gmail_account_id, email_id, job_type) "
        "WHERE email_id IS NOT NULL AND status IN ('PENDING', 'RUNNING')"
    )

    bind.execute(
        AiCategory.__table__.insert(),
        [{"id": uuid4(), "name": name, "description": description, "is_active": True}
         for name, description in CATEGORY_SEEDS],
    )


def downgrade() -> None:
    # Deliberately destructive only when an operator explicitly requests it.
    Base.metadata.drop_all(bind=op.get_bind(), checkfirst=True)

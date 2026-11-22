"""Create the normalized Gmail AI Dashboard schema.

This revision is intentionally additive: it creates tables in the existing
database and never drops/recreates the database itself. It is for an empty
database; an installation with legacy tables must be reconciled deliberately
before applying this initial revision.
"""

from __future__ import annotations

from uuid import uuid4

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


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
    # Frozen from the original revision (dcd8f70). Never import live ORM models
    # here: later revisions own their additional tables and columns.
    op.create_table('ai_categories',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_table('users',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('google_id', sa.String(length=255), nullable=False),
        sa.Column('email', sa.String(length=320), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('picture', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('google_id')
    )
    op.create_table('gmail_accounts',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('email_address', sa.String(length=320), nullable=False),
        sa.Column('provider', sa.String(length=32), server_default='gmail', nullable=False),
        sa.Column('gmail_user_id', sa.String(length=255), nullable=True),
        sa.Column('access_token', sa.Text(), nullable=False),
        sa.Column('refresh_token', sa.Text(), nullable=True),
        sa.Column('token_expiry', sa.DateTime(timezone=True), nullable=True),
        sa.Column('scope', sa.Text(), nullable=True),
        sa.Column('sync_status', sa.String(length=20), server_default='IDLE', nullable=False),
        sa.Column('last_synced_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('history_id', sa.String(length=255), nullable=True),
        sa.Column('ai_classification_enabled', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('id', 'user_id', name='uq_gmail_accounts_id_user'),
        sa.UniqueConstraint('user_id', 'email_address', name='uq_gmail_accounts_user_email')
    )
    op.create_index('ix_gmail_accounts_user_id', 'gmail_accounts', ['user_id'], unique=False)
    op.create_table('emails',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('gmail_account_id', sa.Uuid(), nullable=False),
        sa.Column('gmail_message_id', sa.String(length=255), nullable=False),
        sa.Column('gmail_thread_id', sa.String(length=255), nullable=True),
        sa.Column('sender_email', sa.Text(), nullable=True),
        sa.Column('sender_name', sa.String(length=255), nullable=True),
        sa.Column('recipient_metadata', sa.Text(), nullable=True),
        sa.Column('cc_metadata', sa.Text(), nullable=True),
        sa.Column('bcc_metadata', sa.Text(), nullable=True),
        sa.Column('subject', sa.Text(), nullable=True),
        sa.Column('snippet', sa.Text(), nullable=True),
        sa.Column('body_text', sa.Text(), nullable=True),
        sa.Column('body_html', sa.Text(), nullable=True),
        sa.Column('received_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_read', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('is_starred', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('has_attachments', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('attachments_meta', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
        sa.Column('size_estimate', sa.Integer(), nullable=True),
        sa.Column('history_id', sa.String(length=255), nullable=True),
        sa.Column('raw_metadata', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['gmail_account_id'], ['gmail_accounts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('gmail_account_id', 'gmail_message_id', name='uq_emails_account_message'),
        sa.UniqueConstraint('id', 'gmail_account_id', name='uq_emails_id_account')
    )
    op.create_index('ix_emails_account_received', 'emails', ['gmail_account_id', 'received_at'], unique=False)
    op.create_index('ix_emails_account_thread', 'emails', ['gmail_account_id', 'gmail_thread_id'], unique=False)
    op.create_index('ix_emails_account_unread', 'emails', ['gmail_account_id', 'is_read'], unique=False)
    op.create_table('labels',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('gmail_account_id', sa.Uuid(), nullable=False),
        sa.Column('gmail_label_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('label_type', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['gmail_account_id'], ['gmail_accounts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('gmail_account_id', 'gmail_label_id', name='uq_labels_account_gmail_label'),
        sa.UniqueConstraint('id', 'gmail_account_id', name='uq_labels_id_account')
    )
    op.create_index('ix_labels_account_name', 'labels', ['gmail_account_id', 'name'], unique=False)
    op.create_table('sync_history',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('gmail_account_id', sa.Uuid(), nullable=False),
        sa.Column('sync_type', sa.String(length=30), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('messages_fetched', sa.Integer(), server_default='0', nullable=False),
        sa.Column('messages_inserted', sa.Integer(), server_default='0', nullable=False),
        sa.Column('messages_updated', sa.Integer(), server_default='0', nullable=False),
        sa.Column('labels_synced', sa.Integer(), server_default='0', nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.CheckConstraint("status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED')", name='ck_sync_history_status'),
        sa.ForeignKeyConstraint(['gmail_account_id'], ['gmail_accounts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_sync_history_account_started', 'sync_history', ['gmail_account_id', 'started_at'], unique=False)
    op.create_table('email_classifications',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('email_id', sa.Uuid(), nullable=False),
        sa.Column('category_id', sa.Uuid(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('model_name', sa.String(length=150), nullable=True),
        sa.Column('model_version', sa.String(length=100), nullable=True),
        sa.Column('prompt_version', sa.String(length=100), nullable=True),
        sa.Column('classification_status', sa.String(length=20), server_default='SUCCEEDED', nullable=False),
        sa.Column('classification_reason', sa.Text(), nullable=True),
        sa.Column('is_current', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('classified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.CheckConstraint("classification_status IN ('PENDING', 'SUCCEEDED', 'FAILED')", name='ck_email_classifications_status'),
        sa.CheckConstraint('confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)', name='ck_email_classifications_confidence'),
        sa.ForeignKeyConstraint(['category_id'], ['ai_categories.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_email_classifications_category_id', 'email_classifications', ['category_id'], unique=False)
    op.create_index('ix_email_classifications_email_current', 'email_classifications', ['email_id', 'is_current'], unique=False)
    op.create_table('email_labels',
        sa.Column('email_id', sa.Uuid(), nullable=False),
        sa.Column('label_id', sa.Uuid(), nullable=False),
        sa.Column('gmail_account_id', sa.Uuid(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['email_id', 'gmail_account_id'], ['emails.id', 'emails.gmail_account_id'], name='fk_email_labels_email_account', ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['label_id', 'gmail_account_id'], ['labels.id', 'labels.gmail_account_id'], name='fk_email_labels_label_account', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('email_id', 'label_id')
    )
    op.create_index('ix_email_labels_label_id', 'email_labels', ['label_id'], unique=False)
    op.create_table('llm_classification_jobs',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('gmail_account_id', sa.Uuid(), nullable=False),
        sa.Column('email_id', sa.Uuid(), nullable=True),
        sa.Column('job_type', sa.String(length=30), nullable=False),
        sa.Column('status', sa.String(length=20), server_default='PENDING', nullable=False),
        sa.Column('attempts', sa.Integer(), server_default='0', nullable=False),
        sa.Column('max_attempts', sa.Integer(), server_default='3', nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('emails_discovered', sa.Integer(), server_default='0', nullable=False),
        sa.Column('emails_classified', sa.Integer(), server_default='0', nullable=False),
        sa.Column('emails_remaining', sa.Integer(), server_default='0', nullable=False),
        sa.Column('emails_failed', sa.Integer(), server_default='0', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'PARTIAL')", name='ck_llm_jobs_status'),
        sa.CheckConstraint('attempts >= 0 AND max_attempts > 0 AND attempts <= max_attempts', name='ck_llm_jobs_attempts'),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['gmail_account_id'], ['gmail_accounts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_llm_jobs_account_email', 'llm_classification_jobs', ['gmail_account_id', 'email_id'], unique=False)
    op.create_index('ix_llm_jobs_claim', 'llm_classification_jobs', ['status', 'scheduled_at'], unique=False)

    categories = sa.table(
        "ai_categories", sa.column("id", sa.Uuid()), sa.column("name", sa.String()),
        sa.column("description", sa.Text()), sa.column("is_active", sa.Boolean()),
    )

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

    existing_categories = set(
        bind.execute(sa.select(categories.c.name)).scalars().all()
    )
    missing_seeds = [
        {"id": uuid4(), "name": name, "description": description, "is_active": True}
        for name, description in CATEGORY_SEEDS
        if name not in existing_categories
    ]
    if missing_seeds:
        bind.execute(
            categories.insert(),
            missing_seeds,
        )


def downgrade() -> None:
    # Deliberately destructive only when an operator explicitly requests it.
    op.drop_table("llm_classification_jobs")
    op.drop_table("email_labels")
    op.drop_table("email_classifications")
    op.drop_table("sync_history")
    op.drop_table("labels")
    op.drop_table("emails")
    op.drop_table("gmail_accounts")
    op.drop_table("users")
    op.drop_table("ai_categories")

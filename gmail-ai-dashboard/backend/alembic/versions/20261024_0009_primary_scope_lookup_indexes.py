"""Add composite indexes for tenant-scoped Primary mailbox retrieval."""

from alembic import op


revision = "20261024_0009"
down_revision = "20261024_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Primary scope joins labels by Gmail label ID and then restricts by the
    # account.  The composite indexes avoid scanning every tenant's labels and
    # make label/date retrieval cheaper before vector ranking begins.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_labels_gmail_label_account "
        "ON labels (gmail_label_id, gmail_account_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_email_labels_account_label_email "
        "ON email_labels (gmail_account_id, label_id, email_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_email_labels_account_label_email")
    op.execute("DROP INDEX IF EXISTS ix_labels_gmail_label_account")

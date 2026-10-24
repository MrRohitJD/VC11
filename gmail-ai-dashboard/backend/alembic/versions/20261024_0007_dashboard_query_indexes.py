"""Add the selective index used by dashboard classification lookups.

Revision ID: 20261024_0007
Revises: 20261024_0006
"""

from alembic import op


revision = "20261024_0007"
down_revision = "20261024_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Analytics joins email_classifications by email_id, but only the current
    # successful result is relevant. This keeps historical and in-flight
    # classification rows out of the index.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_email_classifications_dashboard_current "
        "ON email_classifications (email_id) "
        "WHERE is_current = true AND classification_status = 'SUCCEEDED'"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_email_classifications_dashboard_current")

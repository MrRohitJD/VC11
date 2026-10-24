"""Cover dashboard classification lookups with the selected category."""

from alembic import op


revision = "20261024_0008"
down_revision = "20261024_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Dashboard metrics and charts join current successful classifications by
    # email_id and immediately read category_id. Replacing the narrower 0007
    # index keeps the predicate selective while avoiding a heap lookup for the
    # category join on PostgreSQL.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_email_classifications_dashboard_lookup "
        "ON email_classifications (email_id, category_id) "
        "WHERE is_current = true AND classification_status = 'SUCCEEDED'"
    )
    op.execute("DROP INDEX IF EXISTS ix_email_classifications_dashboard_current")


def downgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_email_classifications_dashboard_current "
        "ON email_classifications (email_id) "
        "WHERE is_current = true AND classification_status = 'SUCCEEDED'"
    )
    op.execute("DROP INDEX IF EXISTS ix_email_classifications_dashboard_lookup")

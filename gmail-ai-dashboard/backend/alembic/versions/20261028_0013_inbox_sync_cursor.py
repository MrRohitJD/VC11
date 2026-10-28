"""Persist the cursor for large Inbox backfills."""

from alembic import op
import sqlalchemy as sa


revision = "20261028_0013"
down_revision = "20261024_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("gmail_accounts")}
    if "inbox_sync_page_token" not in columns:
        op.add_column("gmail_accounts", sa.Column("inbox_sync_page_token", sa.Text(), nullable=True))
    if "inbox_sync_heartbeat_at" not in columns:
        op.add_column("gmail_accounts", sa.Column("inbox_sync_heartbeat_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("gmail_accounts")}
    if "inbox_sync_heartbeat_at" in columns:
        op.drop_column("gmail_accounts", "inbox_sync_heartbeat_at")
    if "inbox_sync_page_token" in columns:
        op.drop_column("gmail_accounts", "inbox_sync_page_token")

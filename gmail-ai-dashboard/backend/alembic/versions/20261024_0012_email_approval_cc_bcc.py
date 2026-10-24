"""Add editable CC and BCC recipients to outbound email approvals."""

from alembic import op
import sqlalchemy as sa


revision = "20261024_0012"
down_revision = "20261024_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("email_send_approvals")}
    if "cc" not in columns:
        op.add_column("email_send_approvals", sa.Column("cc", sa.Text(), nullable=True))
    if "bcc" not in columns:
        op.add_column("email_send_approvals", sa.Column("bcc", sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("email_send_approvals")}
    if "bcc" in columns:
        op.drop_column("email_send_approvals", "bcc")
    if "cc" in columns:
        op.drop_column("email_send_approvals", "cc")

"""Add scheduled delivery state and Gmail delivery metadata."""

from alembic import op
import sqlalchemy as sa


revision = "20261024_0010"
down_revision = "20261024_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("email_send_approvals") as batch:
        batch.add_column(sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("delivery_message_id", sa.String(length=255), nullable=True))
        batch.drop_constraint("ck_email_send_approvals_status", type_="check")
        batch.create_check_constraint(
            "ck_email_send_approvals_status",
            "status IN ('PENDING', 'APPROVED', 'SCHEDULED', 'SENDING', 'REJECTED', 'SENT', 'CANCELLED')",
        )
        batch.create_index("ix_email_send_approvals_scheduled", ["status", "scheduled_for"])


def downgrade() -> None:
    with op.batch_alter_table("email_send_approvals") as batch:
        batch.drop_index("ix_email_send_approvals_scheduled")
        batch.drop_constraint("ck_email_send_approvals_status", type_="check")
        batch.create_check_constraint(
            "ck_email_send_approvals_status",
            "status IN ('PENDING', 'APPROVED', 'REJECTED', 'SENT', 'CANCELLED')",
        )
        batch.drop_column("delivery_message_id")
        batch.drop_column("scheduled_for")

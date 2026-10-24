"""Repair email delivery columns when an earlier migration was stamped ahead."""

from alembic import op
import sqlalchemy as sa


revision = "20261024_0011"
down_revision = "20261024_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Make the delivery columns and index present without duplicating them."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("email_send_approvals")}

    if "scheduled_for" not in columns:
        op.add_column(
            "email_send_approvals",
            sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=True),
        )
    if "delivery_message_id" not in columns:
        op.add_column(
            "email_send_approvals",
            sa.Column("delivery_message_id", sa.String(length=255), nullable=True),
        )

    constraint_names = {
        constraint.get("name")
        for constraint in inspector.get_check_constraints("email_send_approvals")
    }
    if "ck_email_send_approvals_status" in constraint_names:
        op.drop_constraint(
            "ck_email_send_approvals_status",
            "email_send_approvals",
            type_="check",
        )
    op.create_check_constraint(
        "ck_email_send_approvals_status",
        "email_send_approvals",
        "status IN ('PENDING', 'APPROVED', 'SCHEDULED', 'SENDING', 'REJECTED', 'SENT', 'CANCELLED')",
    )

    index_names = {index["name"] for index in inspector.get_indexes("email_send_approvals")}
    if "ix_email_send_approvals_scheduled" not in index_names:
        op.create_index(
            "ix_email_send_approvals_scheduled",
            "email_send_approvals",
            ["status", "scheduled_for"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    index_names = {index["name"] for index in inspector.get_indexes("email_send_approvals")}
    if "ix_email_send_approvals_scheduled" in index_names:
        op.drop_index("ix_email_send_approvals_scheduled", table_name="email_send_approvals")

    constraint_names = {
        constraint.get("name")
        for constraint in inspector.get_check_constraints("email_send_approvals")
    }
    if "ck_email_send_approvals_status" in constraint_names:
        op.drop_constraint(
            "ck_email_send_approvals_status",
            "email_send_approvals",
            type_="check",
        )
        op.create_check_constraint(
            "ck_email_send_approvals_status",
            "email_send_approvals",
            "status IN ('PENDING', 'APPROVED', 'REJECTED', 'SENT', 'CANCELLED')",
        )

    columns = {column["name"] for column in inspector.get_columns("email_send_approvals")}
    if "delivery_message_id" in columns:
        op.drop_column("email_send_approvals", "delivery_message_id")
    if "scheduled_for" in columns:
        op.drop_column("email_send_approvals", "scheduled_for")

"""Add shared per-identity chat concurrency leases.

Revision ID: 20261029_0014
Revises: 20261028_0013
"""

from alembic import op
import sqlalchemy as sa


revision = "20261029_0014"
down_revision = "20261028_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_concurrency_locks",
        sa.Column("identity", sa.String(length=80), nullable=False),
        sa.PrimaryKeyConstraint("identity"),
    )
    op.create_table(
        "chat_request_leases",
        sa.Column("lease_id", sa.String(length=64), nullable=False),
        sa.Column("identity", sa.String(length=80), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("lease_id"),
    )
    op.create_index(
        "ix_chat_request_leases_identity_expires_at",
        "chat_request_leases",
        ["identity", "expires_at"],
    )
    op.create_index(
        "ix_chat_request_leases_expires_at",
        "chat_request_leases",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_chat_request_leases_expires_at", table_name="chat_request_leases")
    op.drop_index("ix_chat_request_leases_identity_expires_at", table_name="chat_request_leases")
    op.drop_table("chat_request_leases")
    op.drop_table("chat_concurrency_locks")

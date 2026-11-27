"""Add a per-user cap for Inbox synchronization.

Revision ID: 20261127_0019
Revises: 20261101_0018
Create Date: 2026-11-27 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "20261127_0019"
down_revision = "20261101_0018"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "users",
        sa.Column("mail_sync_limit", sa.Integer(), server_default="5000", nullable=False),
    )
    with op.batch_alter_table("users") as batch_op:
        batch_op.create_check_constraint(
            "ck_users_mail_sync_limit",
            "mail_sync_limit >= 1 AND mail_sync_limit <= 50000",
        )


def downgrade():
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_constraint("ck_users_mail_sync_limit", type_="check")
        batch_op.drop_column("mail_sync_limit")

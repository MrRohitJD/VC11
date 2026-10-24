"""Add durable daily assistant request counters.

Revision ID: 20261024_0005
Revises: 20261023_0004
"""

from alembic import op
import sqlalchemy as sa

revision = "20261024_0005"
down_revision = "20261023_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_quotas",
        sa.Column("identity", sa.String(80), primary_key=True),
        sa.Column("day", sa.Date(), primary_key=True),
        sa.Column("count", sa.Integer(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("chat_quotas")

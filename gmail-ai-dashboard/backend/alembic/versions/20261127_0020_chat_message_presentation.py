"""Persist LLM-designed answer presentations with chat history."""

from alembic import op
import sqlalchemy as sa


revision = "20261127_0020"
down_revision = "20261127_0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_messages", sa.Column("presentation", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_messages", "presentation")

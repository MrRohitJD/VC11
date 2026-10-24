"""Add durable pgvector embeddings and isolated demo accounts.

Revision ID: 20261019_0003
Revises: 20261013_0002
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import HALFVEC


revision = "20261019_0003"
down_revision = "20261013_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.false()))
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        op.add_column("email_embeddings", sa.Column("embedding", HALFVEC(3072), nullable=True))
    else:
        op.add_column("email_embeddings", sa.Column("embedding", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("email_embeddings", "embedding")
    op.drop_column("users", "is_demo")

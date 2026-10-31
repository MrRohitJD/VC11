"""Migrate pgvector embedding storage to 1536 dimensions for text-embedding-3-small.

Revision ID: 20261101_0018
Revises: 20261031_0017
Create Date: 2026-11-01 04:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
try:
    from pgvector.sqlalchemy import HALFVEC
except ImportError:
    HALFVEC = None

revision = "20261101_0018"
down_revision = "20261031_0017"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    dialect_name = bind.dialect.name

    if dialect_name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # 1. Add embedding_1536 column if not present
        op.execute("ALTER TABLE email_embeddings ADD COLUMN IF NOT EXISTS embedding_1536 halfvec(1536)")
        # 2. Add embedding_3072 column if not present
        op.execute("ALTER TABLE email_embeddings ADD COLUMN IF NOT EXISTS embedding_3072 halfvec(3072)")
        # 3. Update defaults for new rows
        op.execute("ALTER TABLE email_embeddings ALTER COLUMN embedding_dimensions SET DEFAULT 1536")
        op.execute("ALTER TABLE email_embeddings ALTER COLUMN vector_collection SET DEFAULT 'postgres_halfvec_1536'")
        op.execute("ALTER TABLE email_embeddings ALTER COLUMN embedding_model SET DEFAULT 'text-embedding-3-small'")
    else:
        # SQLite support for unit tests
        with op.batch_alter_table("email_embeddings") as batch_op:
            batch_op.add_column(sa.Column("embedding_1536", sa.JSON(), nullable=True))
            batch_op.add_column(sa.Column("embedding_3072", sa.JSON(), nullable=True))


def downgrade():
    bind = op.get_bind()
    dialect_name = bind.dialect.name

    if dialect_name == "postgresql":
        op.execute("ALTER TABLE email_embeddings ALTER COLUMN embedding_dimensions SET DEFAULT 3072")
        op.execute("ALTER TABLE email_embeddings ALTER COLUMN vector_collection SET DEFAULT 'postgres_halfvec_3072'")
        op.execute("ALTER TABLE email_embeddings ALTER COLUMN embedding_model SET DEFAULT 'text-embedding-3-large'")
        op.execute("ALTER TABLE email_embeddings DROP COLUMN IF EXISTS embedding_1536")
        op.execute("ALTER TABLE email_embeddings DROP COLUMN IF EXISTS embedding_3072")
    else:
        with op.batch_alter_table("email_embeddings") as batch_op:
            batch_op.drop_column("embedding_3072")
            batch_op.drop_column("embedding_1536")

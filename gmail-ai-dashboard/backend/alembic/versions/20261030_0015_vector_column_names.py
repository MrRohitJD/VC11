"""Rename legacy vector identifiers without rewriting stored embeddings.

Revision ID: 20261030_0015
Revises: 20261029_0014
"""
from alembic import op

revision = "20261030_0015"
down_revision = "20261029_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("email_embeddings", "chroma_collection", new_column_name="vector_collection")
    op.alter_column("email_embeddings", "chroma_record_id", new_column_name="vector_record_id")


def downgrade() -> None:
    op.alter_column("email_embeddings", "vector_record_id", new_column_name="chroma_record_id")
    op.alter_column("email_embeddings", "vector_collection", new_column_name="chroma_collection")

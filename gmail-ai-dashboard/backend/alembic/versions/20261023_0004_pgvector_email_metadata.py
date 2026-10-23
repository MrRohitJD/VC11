"""Add searchable email metadata and pgvector HNSW index.

Revision ID: 20261023_0004
Revises: 20261019_0003
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import HALFVEC

revision = "20261023_0004"
down_revision = "20261019_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    columns = [
        ("email_id", sa.Uuid(), sa.ForeignKey("emails.id", ondelete="CASCADE")),
        ("email_subject", sa.Text(), None),
        ("sender_email", sa.Text(), None),
        ("recipient_emails", sa.ARRAY(sa.Text()), None),
        ("email_sent_at", sa.DateTime(timezone=True), None),
        ("embedding_text", sa.Text(), None),
        ("embedding_type", sa.String(30), None),
    ]
    for name, column_type, foreign_key in columns:
        if is_postgres:
            exists = bind.execute(sa.text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'email_embeddings' AND column_name = :name"
            ), {"name": name}).scalar()
        else:
            exists = False
        if not exists:
            args = [foreign_key] if foreign_key is not None else []
            kwargs = {"server_default": "email"} if name == "embedding_type" else {}
            op.add_column("email_embeddings", sa.Column(name, column_type, *args, nullable=True, **kwargs))

    if is_postgres:
        op.execute("UPDATE email_embeddings SET embedding_type = 'email' WHERE embedding_type IS NULL")
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_email_embeddings_embedding_hnsw "
            "ON email_embeddings USING hnsw (embedding halfvec_cosine_ops)"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_email_embeddings_embedding_hnsw")
    for name in ["embedding_type", "embedding_text", "email_sent_at", "recipient_emails", "sender_email", "email_subject", "email_id"]:
        op.drop_column("email_embeddings", name)

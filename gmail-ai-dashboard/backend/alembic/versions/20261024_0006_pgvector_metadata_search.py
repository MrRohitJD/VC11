"""Backfill pgvector metadata and add searchable metadata indexes.

Revision ID: 20261024_0006
Revises: 20261024_0005
"""

from alembic import op
import sqlalchemy as sa


revision = "20261024_0006"
down_revision = "20261024_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    # Populate the metadata columns for rows created before the metadata
    # migration. Their vector is marked OUTDATED below when it is absent or
    # has the old 384-dimensional shape, so the indexer will regenerate it.
    op.execute(sa.text("""
        UPDATE email_embeddings AS ee
        SET email_id = ec.email_id,
            email_subject = COALESCE(e.subject, ''),
            sender_email = COALESCE(e.sender_email, ''),
            recipient_emails = CASE
                WHEN NULLIF(TRIM(COALESCE(e.recipient_metadata, '')), '') IS NULL THEN ARRAY[]::text[]
                ELSE regexp_split_to_array(e.recipient_metadata, '\\s*,\\s*')
            END,
            email_sent_at = e.received_at,
            embedding_text = COALESCE(ee.embedding_text, ec.chunk_text),
            embedding_type = COALESCE(ee.embedding_type, 'email')
        FROM email_chunks AS ec
        LEFT JOIN emails AS e ON e.id = ec.email_id
        WHERE ee.chunk_id = ec.id
    """))
    op.execute(sa.text("""
        UPDATE email_embeddings
        SET status = 'OUTDATED'
        WHERE embedding IS NULL OR embedding_dimensions <> 3072
    """))
    op.execute("ALTER TABLE email_embeddings ALTER COLUMN embedding_dimensions SET DEFAULT 3072")
    op.execute("ALTER TABLE email_embeddings ALTER COLUMN chroma_collection SET DEFAULT 'postgres_halfvec_3072'")

    op.execute("CREATE INDEX IF NOT EXISTS ix_email_embeddings_email_id ON email_embeddings (email_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_email_embeddings_sender_email ON email_embeddings (sender_email)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_email_embeddings_email_sent_at ON email_embeddings (email_sent_at)")
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_email_embeddings_metadata_fts
        ON email_embeddings USING gin (
            to_tsvector('simple',
                coalesce(email_subject, '') || ' ' ||
                coalesce(sender_email, '') || ' ' ||
                coalesce(embedding_text, '')
            )
        )
    """)


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_email_embeddings_metadata_fts")
    op.execute("DROP INDEX IF EXISTS ix_email_embeddings_email_sent_at")
    op.execute("DROP INDEX IF EXISTS ix_email_embeddings_sender_email")
    op.execute("DROP INDEX IF EXISTS ix_email_embeddings_email_id")

"""Store encrypted per-account developer connections and provider keys."""

from alembic import op
import sqlalchemy as sa

revision = "20261030_0016"
down_revision = "20261030_0015"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("developer_database_url", sa.Text(), nullable=True))
    op.add_column("users", sa.Column("developer_llm_provider", sa.String(20), nullable=True))
    op.add_column("users", sa.Column("developer_llm_api_key", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("users", "developer_llm_api_key")
    op.drop_column("users", "developer_llm_provider")
    op.drop_column("users", "developer_database_url")

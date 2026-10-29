"""The identifier migration must preserve vectors and uniqueness in both directions."""
import importlib.util
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
import sqlalchemy as sa


def test_vector_identifier_migration_preserves_rows_and_constraints():
    path = Path(__file__).parents[1] / "gmail-ai-dashboard/backend/alembic/versions/20261030_0015_vector_column_names.py"
    spec = importlib.util.spec_from_file_location("vector_column_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.exec_driver_sql('''CREATE TABLE email_embeddings (
            chunk_id TEXT NOT NULL,
            chroma_collection TEXT NOT NULL,
            chroma_record_id TEXT NOT NULL,
            embedding TEXT,
            UNIQUE (chroma_collection, chroma_record_id),
            UNIQUE (chunk_id, chroma_collection)
        )''')
        connection.exec_driver_sql("INSERT INTO email_embeddings VALUES ('chunk', 'postgres_halfvec_3072', 'record', '[1, 2]')")
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
        assert connection.exec_driver_sql("SELECT vector_collection, vector_record_id, embedding FROM email_embeddings").one() == ("postgres_halfvec_3072", "record", "[1, 2]")
        with pytest.raises(sa.exc.IntegrityError):
            connection.exec_driver_sql("INSERT INTO email_embeddings VALUES ('chunk', 'postgres_halfvec_3072', 'other', '[3, 4]')")
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
        assert connection.exec_driver_sql("SELECT chroma_record_id, embedding FROM email_embeddings").one() == ("record", "[1, 2]")
    engine.dispose()

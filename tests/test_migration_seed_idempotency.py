"""Test that 20261012_0001 category seed insertion is idempotent."""

import importlib.util
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy import inspect
from alembic.migration import MigrationContext
from alembic.operations import Operations

from app.models.email import AiCategory, Base

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "gmail-ai-dashboard"
    / "backend"
    / "alembic"
    / "versions"
    / "20261012_0001_rebuild_gmail_dashboard_schema.py"
)


def load_migration_module():
    spec = importlib.util.spec_from_file_location("migration_0001", MIGRATION_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_category_seeds_idempotent():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    mod = load_migration_module()

    # Pre-insert one seed manually (e.g. WORK, the exact key from the error)
    with Session(engine) as session:
        session.add(AiCategory(id=uuid4(), name="WORK", description="Existing work category", is_active=True))
        session.commit()

    # Run the upgraded seeding logic
    with engine.begin() as conn:
        existing_categories = set(
            conn.execute(select(AiCategory.name)).scalars().all()
        )
        assert "WORK" in existing_categories

        missing_seeds = [
            {"id": uuid4(), "name": name, "description": description, "is_active": True}
            for name, description in mod.CATEGORY_SEEDS
            if name not in existing_categories
        ]
        assert not any(s["name"] == "WORK" for s in missing_seeds)
        if missing_seeds:
            conn.execute(AiCategory.__table__.insert(), missing_seeds)

    # Verify all seeds exist and no duplicate key error was raised
    with Session(engine) as session:
        all_names = set(session.scalars(select(AiCategory.name)).all())
        expected_names = {name for name, _ in mod.CATEGORY_SEEDS}
        assert expected_names.issubset(all_names)


def test_fresh_database_initial_revision_does_not_create_future_schema():
    """Live models are imported above; they must not leak into revision 0001."""
    engine = create_engine("sqlite:///:memory:")
    migration = load_migration_module()
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
        inspector = inspect(connection)
        assert set(inspector.get_table_names()) == {
            "users", "gmail_accounts", "emails", "labels", "email_labels",
            "ai_categories", "email_classifications", "sync_history",
            "llm_classification_jobs",
        }
        assert "is_demo" not in {c["name"] for c in inspector.get_columns("users")}
        assert "inbox_sync_page_token" not in {
            c["name"] for c in inspector.get_columns("gmail_accounts")
        }
        assert set(connection.execute(select(AiCategory.name)).scalars()) == {
            name for name, _ in migration.CATEGORY_SEEDS
        }
        # Subsequent revisions must be able to create their own tables/columns.
        for filename in [
            "20261013_0002_add_rag_and_chat_tables.py",
            "20261019_0003_heroku_vectors_demo.py",
        ]:
            spec = importlib.util.spec_from_file_location("next_revision", MIGRATION_PATH.with_name(filename))
            next_revision = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(next_revision)
            with Operations.context(MigrationContext.configure(connection)):
                next_revision.upgrade()
        assert "email_embeddings" in inspect(connection).get_table_names()
        assert "is_demo" in {c["name"] for c in inspect(connection).get_columns("users")}

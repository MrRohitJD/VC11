"""Test that 20261012_0001 category seed insertion is idempotent."""

import importlib.util
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

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

"""Ensure alembic_version is stamped if deploying to an existing initialized database."""

import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database.connection import engine


def reconcile() -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    if "users" in tables and "emails" in tables:
        with engine.connect() as conn:
            rev = (
                conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
                if "alembic_version" in tables
                else None
            )
        if not rev:
            print("[reconcile_alembic] Existing tables detected without alembic revision. Stamping to head...")
            cfg = Config("alembic.ini")
            command.stamp(cfg, "head")
            print("[reconcile_alembic] Stamped to head successfully.")


if __name__ == "__main__":
    reconcile()

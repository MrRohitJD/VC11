"""Prepare the fictional demo account after applying Alembic migrations.

Run from backend: python scripts/seed_demo_mailbox.py
Use --check to inspect schema readiness without writing.
"""

import argparse
import sys
from pathlib import Path

from sqlalchemy import func, inspect, select, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database.connection import SessionLocal, engine  # noqa: E402
from app.models.email import Email, EmailClassification, EmailLabel, GmailAccount, Label, User  # noqa: E402
from app.models.rag import EmailChunk, EmailEmbedding  # noqa: E402
from app.services.demo_seed import AI_DESCRIPTIONS, DEMO_EMAIL, GMAIL_CATEGORIES, SAMPLE_MAIL, seed_demo_mailbox  # noqa: E402


REQUIRED_TABLES = {
    "users", "gmail_accounts", "emails", "labels", "email_labels",
    "ai_categories", "email_classifications",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="inspect the database without writing")
    parser.add_argument("--verify", action="store_true", help="verify the seeded mailbox without writing")
    args = parser.parse_args()

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    missing = REQUIRED_TABLES - tables
    has_demo_column = "users" in tables and any(
        column["name"] == "is_demo" for column in inspector.get_columns("users")
    )
    with engine.connect() as connection:
        revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar() if "alembic_version" in tables else None
        vector_available = connection.execute(text(
            "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'vector')"
        )).scalar() if engine.dialect.name == "postgresql" else None
        existing_demo = connection.execute(text(
            "SELECT count(*) FROM users WHERE email = :email"
        ), {"email": DEMO_EMAIL}).scalar() if "users" in tables else 0
    print({"missing_tables": sorted(missing), "has_demo_column": has_demo_column,
           "alembic_revision": revision, "vector_extension_available": vector_available,
           "demo_account_exists": bool(existing_demo)})
    if args.check:
        return
    if missing or not has_demo_column:
        raise SystemExit("Apply the database migrations before seeding the demo mailbox")

    with SessionLocal() as db:
        try:
            user = db.scalar(select(User).where(User.email == DEMO_EMAIL)) if args.verify else seed_demo_mailbox(db)
            if user is None:
                raise RuntimeError("Demo mailbox has not been seeded")
            account = db.scalar(select(GmailAccount).where(GmailAccount.user_id == user.id))
            message_count = db.scalar(select(func.count(Email.id)).where(
                Email.gmail_account_id == account.id,
                Email.gmail_message_id.like("synthetic-demo-%"))) or 0
            category_ids = {row[0] for row in db.execute(
                select(Label.label_id).join(EmailLabel, EmailLabel.label_id == Label.id)
                .where(EmailLabel.gmail_account_id == account.id,
                       Label.label_id.like("CATEGORY_%"))).all()}
            ai_label_count = db.scalar(select(func.count(Label.id)).where(
                Label.gmail_account_id == account.id, Label.type == "ai")) or 0
            classification_count = db.scalar(select(func.count(EmailClassification.id))
                .join(Email, Email.id == EmailClassification.email_id)
                .where(Email.gmail_account_id == account.id,
                       EmailClassification.is_current.is_(True))) or 0
            chunk_count = db.scalar(select(func.count(EmailChunk.id))
                .join(Email, Email.id == EmailChunk.email_id)
                .where(Email.gmail_account_id == account.id)) or 0
            embedding_count = db.scalar(select(func.count(EmailEmbedding.id))
                .join(EmailChunk, EmailChunk.id == EmailEmbedding.chunk_id)
                .join(Email, Email.id == EmailChunk.email_id)
                .where(Email.gmail_account_id == account.id)) or 0
            summary = {"demo_email": user.email, "synthetic_messages": message_count,
                       "gmail_categories": sorted(category_ids),
                       "ai_labels": ai_label_count,
                       "ai_classifications": classification_count,
                       "chunks": chunk_count, "embeddings": embedding_count}
            print(summary)
            if (message_count != len(SAMPLE_MAIL)
                    or category_ids != {key for key, _ in GMAIL_CATEGORIES}
                    or ai_label_count != len(AI_DESCRIPTIONS)
                    or classification_count != len(SAMPLE_MAIL)
                    or chunk_count or embedding_count
                    or account.provider != "demo" or account.access_token):
                raise RuntimeError("Demo mailbox verification failed")
        except Exception:
            db.rollback()
            raise


if __name__ == "__main__":
    main()

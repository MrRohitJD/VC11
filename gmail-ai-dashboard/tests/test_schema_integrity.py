"""Database mapping and integrity tests for the rebuilt core schema."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

import os
import sys

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.models.email import Base, Email, EmailLabel, GmailAccount, Label, User
import app.models.llm  # noqa: F401 - include the ninth core table


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session, engine
    session.close()
    engine.dispose()


def test_all_nine_core_tables_and_account_scoped_message_key(db):
    session, engine = db
    expected = {
        "users", "gmail_accounts", "emails", "labels", "email_labels",
        "ai_categories", "email_classifications", "sync_history",
        "llm_classification_jobs",
    }
    assert expected <= set(inspect(engine).get_table_names())

    user = User(google_id="schema-user", email="schema@example.com")
    session.add(user)
    session.flush()
    account_a = GmailAccount(user_id=user.id, email="a@example.com", access_token="token")
    account_b = GmailAccount(user_id=user.id, email="b@example.com", access_token="token")
    session.add_all([account_a, account_b])
    session.flush()
    # Gmail can issue the same-looking ID in distinct account namespaces.
    session.add_all([
        Email(gmail_account_id=account_a.id, gmail_message_id="message-1"),
        Email(gmail_account_id=account_b.id, gmail_message_id="message-1"),
    ])
    session.commit()

    session.add(Email(gmail_account_id=account_a.id, gmail_message_id="message-1"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_email_label_composite_foreign_keys_reject_cross_account_mapping(db):
    session, _ = db
    user = User(google_id="integrity-user", email="integrity@example.com")
    session.add(user)
    session.flush()
    account_a = GmailAccount(user_id=user.id, email="one@example.com", access_token="token")
    account_b = GmailAccount(user_id=user.id, email="two@example.com", access_token="token")
    session.add_all([account_a, account_b])
    session.flush()
    email = Email(gmail_account_id=account_a.id, gmail_message_id=str(uuid4()))
    foreign_label = Label(gmail_account_id=account_b.id, label_id="INBOX", name="Inbox")
    session.add_all([email, foreign_label])
    session.commit()

    session.add(EmailLabel(
        email_id=email.id,
        label_id=foreign_label.id,
        gmail_account_id=account_a.id,
    ))
    with pytest.raises(IntegrityError):
        session.commit()

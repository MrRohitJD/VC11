"""Tests for listing sent emails with pagination and search."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.routes.emails import list_emails
from app.models.email import Base, Email, EmailLabel, GmailAccount, Label, User


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_list_emails_sent_only_filtering(test_db):
    user = User(id=uuid4(), google_id="user-1", email="me@example.com", is_demo=True)
    test_db.add(user)
    test_db.flush()

    account = GmailAccount(
        id=uuid4(),
        user_id=user.id,
        email="me@example.com",
        access_token="tok",
        sync_status="COMPLETED",
    )
    test_db.add(account)
    test_db.flush()

    sent_label = Label(id=uuid4(), gmail_account_id=account.id, label_id="SENT", name="SENT", type="system")
    inbox_label = Label(id=uuid4(), gmail_account_id=account.id, label_id="INBOX", name="INBOX", type="system")
    test_db.add_all([sent_label, inbox_label])
    test_db.flush()

    # Create 1 inbox email and 2 sent emails
    inbox_email = Email(
        id=uuid4(),
        gmail_account_id=account.id,
        gmail_message_id="msg-inbox-1",
        from_email="other@example.com",
        to_email="me@example.com",
        subject="Incoming Newsletter",
        snippet="Here is your daily news.",
        received_at=datetime(2026, 10, 25, 10, 0, tzinfo=timezone.utc),
        is_read=True,
    )
    sent_email_1 = Email(
        id=uuid4(),
        gmail_account_id=account.id,
        gmail_message_id="msg-sent-1",
        from_email="me@example.com",
        to_email="partner@example.com",
        subject="Project Update",
        snippet="Here is the latest progress report.",
        received_at=datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc),
        is_read=True,
    )
    sent_email_2 = Email(
        id=uuid4(),
        gmail_account_id=account.id,
        gmail_message_id="msg-sent-2",
        from_email="me@example.com",
        to_email="recruiter@example.com",
        subject="Interview availability",
        snippet="I am available on Thursday morning.",
        received_at=datetime(2026, 10, 27, 14, 0, tzinfo=timezone.utc),
        is_read=True,
    )
    test_db.add_all([inbox_email, sent_email_1, sent_email_2])
    test_db.flush()

    # Link labels
    test_db.add(EmailLabel(email_id=inbox_email.id, label_id=inbox_label.id, gmail_account_id=account.id))
    test_db.add(EmailLabel(email_id=sent_email_1.id, label_id=sent_label.id, gmail_account_id=account.id))
    test_db.add(EmailLabel(email_id=sent_email_2.id, label_id=sent_label.id, gmail_account_id=account.id))
    test_db.commit()

    # Query with sent_only=True
    res = list_emails(
        page=1,
        per_page=10,
        sent_only=True,
        user=user,
        db=test_db,
    )

    assert res["total"] == 2
    assert len(res["emails"]) == 2
    subjects = [e["subject"] for e in res["emails"]]
    assert "Project Update" in subjects
    assert "Interview availability" in subjects
    assert "Incoming Newsletter" not in subjects

    # Search filtering by recipient
    search_res = list_emails(
        page=1,
        per_page=10,
        sent_only=True,
        search="recruiter",
        user=user,
        db=test_db,
    )
    assert search_res["total"] == 1
    assert search_res["emails"][0]["subject"] == "Interview availability"

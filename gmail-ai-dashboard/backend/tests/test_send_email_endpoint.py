"""Unit tests for the send email endpoint."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.routes.emails import SendEmailRequest, send_email_endpoint
from app.models.email import Base, Email, GmailAccount, Label, User


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_send_email_missing_recipient(test_db):
    user = User(id=uuid4(), google_id="user1", email="user@example.com", is_demo=False)
    test_db.add(user)
    test_db.commit()

    req = SendEmailRequest(to="", subject="Hello", body="World")
    with pytest.raises(HTTPException) as exc_info:
        send_email_endpoint(req, user=user, db=test_db)
    assert exc_info.value.status_code == 422
    assert "recipient" in str(exc_info.value.detail).lower()


def test_send_email_invalid_recipient_format(test_db):
    user = User(id=uuid4(), google_id="user1", email="user@example.com", is_demo=False)
    test_db.add(user)
    test_db.commit()

    req = SendEmailRequest(to="not-an-email", subject="Hello", body="World")
    with pytest.raises(HTTPException) as exc_info:
        send_email_endpoint(req, user=user, db=test_db)
    assert exc_info.value.status_code == 422


def test_send_email_demo_mode_simulation(test_db):
    user = User(id=uuid4(), google_id="demo-user", email="demo@example.com", is_demo=True)
    test_db.add(user)
    test_db.flush()

    account = GmailAccount(
        id=uuid4(),
        user_id=user.id,
        email="demo@example.com",
        access_token="fake-token",
        sync_status="COMPLETED",
    )
    test_db.add(account)
    test_db.commit()

    req = SendEmailRequest(to="friend@example.com", subject="Test Demo", body="Demo body")
    response = send_email_endpoint(req, user=user, db=test_db)

    assert response["status"] == "SENT"
    assert response["demo"] is True
    assert "demo" in response["message"].lower()

    # Verify email was saved in DB with SENT label
    saved = test_db.query(Email).filter_by(gmail_account_id=account.id).first()
    assert saved is not None
    assert saved.to_email == "friend@example.com"
    assert saved.subject == "Test Demo"
    assert saved.body == "Demo body"


def test_send_email_authenticated_user_success(test_db):
    user = User(id=uuid4(), google_id="real-user", email="me@example.com", is_demo=False)
    test_db.add(user)
    test_db.flush()

    account = GmailAccount(
        id=uuid4(),
        user_id=user.id,
        email="me@example.com",
        access_token="valid-token",
        scope="https://www.googleapis.com/auth/gmail.send",
        sync_status="COMPLETED",
    )
    test_db.add(account)
    test_db.commit()

    req = SendEmailRequest(
        to="recipient@example.com",
        subject="Meeting tomorrow",
        body="Let's meet at 10 AM",
        cc="colleague@example.com",
    )

    with patch("app.api.routes.emails.send_gmail_message", return_value={"id": "msg-12345", "threadId": "t-1"}) as mock_send:
        response = send_email_endpoint(req, user=user, db=test_db)

        assert response["status"] == "SENT"
        assert response["delivery_message_id"] == "msg-12345"
        assert response["demo"] is False
        mock_send.assert_called_once()
        _, kwargs = mock_send.call_args
        assert kwargs["recipient"] == "recipient@example.com"
        assert kwargs["subject"] == "Meeting tomorrow"
        assert kwargs["body"] == "Let's meet at 10 AM"
        assert kwargs["cc"] == "colleague@example.com"

"""Verify outbound MIME headers for approved Gmail delivery."""

from base64 import urlsafe_b64decode
from email import policy
from email.parser import BytesParser
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.email import Base, GmailAccount, User
from app.services.gmail_service import GMAIL_SEND_SCOPE, send_gmail_message


def test_send_gmail_message_includes_cc_and_bcc_headers():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    user = User(id=uuid4(), google_id="gmail-header-test", email="owner@example.com", name="Owner")
    account = GmailAccount(
        user_id=user.id,
        email="owner@example.com",
        access_token="token",
        scope=GMAIL_SEND_SCOPE,
    )
    db.add_all([user, account])
    db.commit()

    class FakeRequest:
        def __init__(self, service):
            self.service = service

        def execute(self):
            return {"id": "gmail-message-cc-bcc"}

    class FakeGmailService:
        def __init__(self):
            self.raw = None

        def users(self):
            return self

        def messages(self):
            return self

        def send(self, *, userId, body):
            self.raw = body["raw"]
            return FakeRequest(self)

    service = FakeGmailService()
    with patch("app.services.gmail_service.build_gmail_service_for_user", return_value=service):
        result = send_gmail_message(
            db,
            user,
            "recipient@example.com",
            "Subject",
            "Message body",
            "advisor@example.com, mentor@example.com",
            "archive@example.com",
        )

    message = BytesParser(policy=policy.default).parsebytes(urlsafe_b64decode(service.raw))
    assert result["id"] == "gmail-message-cc-bcc"
    assert message["To"] == "recipient@example.com"
    assert message["Cc"] == "advisor@example.com, mentor@example.com"
    assert message["Bcc"] == "archive@example.com"

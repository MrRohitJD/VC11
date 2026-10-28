import json
from datetime import datetime, timedelta, timezone

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import app.models.llm  # Register related ORM tables for the in-memory database.
from app.models.email import Base, Email, EmailLabel, GmailAccount, Label, User

from app.rag import classification_query
from app.rag.adaptive import plan_retrieval
from app.rag.chains import GroundedRAGChain
from app.rag.query_constraints import DateRangeError, mailbox_time_context, received_date_bounds
from app.rag.request_interpreter import interpret_request


NOW = datetime(2026, 10, 28, 15, 26, tzinfo=timezone(timedelta(hours=5, minutes=30)))


@pytest.mark.parametrize("query,start,end", [
    ("recent emails", "2026-09-28T09:56:00+00:00", "2026-10-28T09:56:00+00:00"),
    ("last two hours", "2026-10-28T07:56:00+00:00", "2026-10-28T09:56:00+00:00"),
    ("last week", "2026-10-18T18:30:00+00:00", "2026-10-25T18:30:00+00:00"),
    ("last quarter", "2026-06-30T18:30:00+00:00", "2026-09-30T18:30:00+00:00"),
    ("past 2 quarters", "2026-04-28T09:56:00+00:00", "2026-10-28T09:56:00+00:00"),
    ("in October 2025", "2025-09-30T18:30:00+00:00", "2025-10-31T18:30:00+00:00"),
    ("in 2024", "2023-12-31T18:30:00+00:00", "2024-12-31T18:30:00+00:00"),
    ("between 2025-12-31 and 2026-01-02", "2025-12-30T18:30:00+00:00", "2026-01-02T18:30:00+00:00"),
    ("since 2026-10-01", "2026-09-30T18:30:00+00:00", "2026-10-28T09:56:00+00:00"),
    ("before 2026-10-01", "1969-12-31T18:30:00+00:00", "2026-09-30T18:30:00+00:00"),
    ("from November 2025 to February 2026", "2025-10-31T18:30:00+00:00", "2026-02-28T18:30:00+00:00"),
    ("between 2023 and 2024", "2022-12-31T18:30:00+00:00", "2024-12-31T18:30:00+00:00"),
])
def test_receipt_windows_cover_calendar_and_rolling_ranges(query, start, end):
    with mailbox_time_context("Asia/Kolkata"):
        bounds = received_date_bounds(query, NOW)
    assert tuple(value.isoformat() for value in bounds) == (start, end)


@pytest.mark.parametrize("query", [
    "between 2026-11-01 and 2026-10-01",
    "on 2026-02-30",
    "last 0 days",
    "between 2027 and 2026",
])
def test_invalid_ranges_ask_for_clarification_instead_of_searching_everything(query):
    with pytest.raises(DateRangeError):
        received_date_bounds(query, NOW)
    request = interpret_request(None, f"Classify emails {query}")
    assert request.intent == "CLARIFY"
    assert request.clarification_question


def test_unsupported_classification_timeframe_is_not_silently_unbounded():
    request = interpret_request(None, "Classify emails from the last 5 business days")
    assert request.intent == "CLARIFY"
    request = interpret_request(None, "Classify emails from 01/10/2026 to 28/10/2026")
    assert request.intent == "CLARIFY"


def test_zero_email_limit_asks_for_clarification():
    request = interpret_request(None, "Classify my latest 0 emails")
    assert request.intent == "CLARIFY"
    assert "greater than zero" in request.clarification_question


def test_polite_classification_request_skips_interpreter_provider():
    request = interpret_request(None, "Could you please analyze my recent emails and categorize them as Work or Important?")
    assert request.intent == "RAG_QUERY"


def test_calendar_day_respects_daylight_saving_boundary():
    with mailbox_time_context("America/New_York"):
        start, end = received_date_bounds("emails received today", datetime(2026, 3, 8, 15, tzinfo=timezone.utc))
    assert start == datetime(2026, 3, 8, 5, tzinfo=timezone.utc)
    assert end == datetime(2026, 3, 9, 4, tzinfo=timezone.utc)


def _documents(count):
    return [Document(page_content=f"Message body {index}", metadata={
        "email_id": str(index), "subject": f"Subject {index}",
        "sender": "sender@example.com", "received_at": "2026-10-28T09:00:00+00:00",
    }) for index in range(count)]


class ClassificationModel:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def invoke(self, messages):
        self.calls += 1
        if self.calls == self.fail_at:
            raise TimeoutError("model unavailable")
        emails = json.loads(messages[-1].content)
        return AIMessage(content=json.dumps({"classifications": [
            {"email_id": item["email_id"], "category": "Work", "reason": "A work update."}
            for item in emails
        ]}))


def test_classification_uses_every_scoped_email_in_bounded_batches(monkeypatch):
    seen = []
    monkeypatch.setattr(classification_query, "scoped_mailbox_documents",
                        lambda db, uid, query, filters: seen.append((query, filters)) or _documents(17))
    monkeypatch.setattr(classification_query, "hydrate_email_documents",
                        lambda docs, db, uid, full_body: docs)
    model = ClassificationModel()
    query = "Analyze my recent emails and categorize them as Work, Job Opportunities, Education, Promotions, Social, or Important. Explain each classification briefly."
    answer, citations, coverage = classification_query.classify_mailbox_query(None, "owner", query, model)
    assert model.calls == 3
    assert len(citations) == 17
    assert "Classified 17 of 17" in answer
    assert "**Work**. A work update." in answer
    assert seen == [(query, {})]
    assert coverage == {"complete": True, "total": 17, "error": None}
    assert plan_retrieval(query, mcp_enabled=False).exhaustive
    assert classification_query.requested_categories(query) == classification_query.DEFAULT_CATEGORIES


def test_classification_timeout_reports_partial_coverage_without_guessing(monkeypatch):
    monkeypatch.setattr(classification_query, "scoped_mailbox_documents",
                        lambda *args: _documents(17))
    monkeypatch.setattr(classification_query, "hydrate_email_documents",
                        lambda docs, db, uid, full_body: docs)
    answer, citations, coverage = classification_query.classify_mailbox_query(
        None, "owner", "Classify emails in October 2026", ClassificationModel(fail_at=2))
    assert len(citations) == 8
    assert "remaining 9 emails were not classified" in answer
    assert "time limit" in answer
    assert coverage == {"complete": False, "total": 17, "error": "timeout"}


def test_explicit_message_count_limits_classification_within_requested_timeframe(monkeypatch):
    monkeypatch.setattr(classification_query, "scoped_mailbox_documents",
                        lambda *args: _documents(17))
    monkeypatch.setattr(classification_query, "hydrate_email_documents",
                        lambda docs, db, uid, full_body: docs)
    answer, citations, coverage = classification_query.classify_mailbox_query(
        None, "owner", "Classify the latest 3 emails from October 2026", ClassificationModel())
    assert len(citations) == 3
    assert coverage["total"] == 3
    assert "Classified 3 of 3" in answer
    assert classification_query.requested_categories(
        "Classify emails as Work or Important from last month") == ("Work", "Important")


def test_classification_scope_is_account_bound_and_keeps_requested_dates_with_bill_category(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "gmail_mcp_enabled", False)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        owners = [User(google_id=f"classification-{i}", email=f"owner-{i}@example.com") for i in range(2)]
        db.add_all(owners)
        db.flush()
        accounts = [GmailAccount(user_id=owner.id, email=owner.email, access_token="test") for owner in owners]
        db.add_all(accounts)
        db.flush()
        spam_label = Label(gmail_account_id=accounts[0].id, label_id="SPAM", name="SPAM")
        db.add(spam_label)
        db.flush()

        def add(message_id, account, month, spam=False):
            email = Email(gmail_account_id=account.id, gmail_message_id=message_id,
                          subject=message_id, body="A work update.", from_email="person@example.com",
                          received_at=datetime(2026, month, 15, tzinfo=timezone.utc))
            db.add(email)
            db.flush()
            if spam:
                db.add(EmailLabel(email_id=email.id, label_id=spam_label.id,
                                  gmail_account_id=account.id))
            return email

        expected = add("owner-october", accounts[0], 10)
        add("owner-september", accounts[0], 9)
        add("other-october", accounts[1], 10)
        add("owner-spam", accounts[0], 10, spam=True)
        db.commit()
        answer, citations, coverage = classification_query.classify_mailbox_query(
            db, owners[0].id,
            "Classify emails from October 2026 as Work, Bills, Promotions",
            ClassificationModel(),
        )
        assert len(citations) == 1
        assert citations[0].email_id == str(expected.id)
        assert "Classified 1 of 1" in answer
        assert coverage["complete"]
        chain = GroundedRAGChain.__new__(GroundedRAGChain)
        chain.model = ClassificationModel()
        result = chain.answer_query(db, "Classify emails from October 2026 as Work, Bills, Promotions",
                                    owners[0].id, top_k=1)
        assert result["retrieval_strategy"] == "exhaustive_mailbox_classification"
        assert result["context_count"] == 1
        assert result["relevance_validated"] is True
        assert "synced Gmail messages" in result["answer"]
        from app.rag import gmail_mcp
        monkeypatch.setattr(settings, "gmail_mcp_enabled", True)
        monkeypatch.setattr(gmail_mcp, "search_live_mailbox", lambda *a, **k: {
            "email_ids": [str(expected.id)], "complete": False, "error_code": "timeout",
        })
        partial = chain.answer_query(db, "Classify emails from October 2026 as Work, Bills, Promotions",
                                     owners[0].id, top_k=1)
        assert partial["context_count"] == 1
        assert partial["relevance_validated"] is False
        assert "additional matching emails may be missing" in partial["answer"]
    engine.dispose()

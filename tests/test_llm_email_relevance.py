import json
from datetime import datetime, timezone

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.rag.query_constraints import received_date_bounds, mailbox_time_context
from app.rag.relevance import validate_email_relevance


def candidate(eid="invitation"):
    return Document(page_content="Your technical interview is confirmed for 23 October 2026 at 10 AM IST.", metadata={
        "email_id": eid, "gmail_message_id": "gmail-123", "subject": "Technical interview confirmation",
        "sender_email": "recruiter@example.com", "sender_name": "Recruiter",
        "received_at": "2026-10-19T09:00:00+00:00",
    })


def decision(eid="invitation", relevant=True):
    return {"email_id": eid, "relevant": relevant, "reason": "Direct invitation for the requested interview." if relevant else "Generic advertisement.",
            "key_points": ["Your technical interview is on 23 October 2026 at 10 AM IST."] if relevant else []}


def test_current_month_has_exclusive_local_calendar_boundaries():
    with mailbox_time_context("Asia/Kolkata"):
        start, end = received_date_bounds("Find all interview related emails of this month", datetime(2026, 10, 19))
    assert start == datetime(2026, 9, 30, 18, 30, tzinfo=timezone.utc)
    assert end == datetime(2026, 10, 31, 18, 30, tzinfo=timezone.utc)


def test_payment_due_month_is_not_interpreted_as_received_month():
    assert received_date_bounds("What are my pending bills due this month?") is None


def test_llm_receives_query_full_content_and_envelope_then_selects_sources():
    observed = []
    def respond(messages):
        observed.append(json.loads(messages[-1].content))
        return AIMessage(content=json.dumps({"decisions": [decision(), decision("job-ad", False)]}))
    job = Document(page_content="HR responsibilities include coordinating interview schedules.", metadata={
        "email_id": "job-ad", "subject": "HR internship advertisement", "sender_email": "jobs@example.com",
        "received_at": "2026-10-18T09:00:00+00:00",
    })
    query = "Find all interview related emails of this month"
    result = validate_email_relevance([candidate(), job], query, RunnableLambda(respond))
    assert not result.failed
    assert [doc.metadata["email_id"] for doc in result.documents] == ["invitation"]
    assert result.documents[0].metadata["relevance_validated"] is True
    payload = observed[0]
    assert payload["query"] == query
    actual = payload["candidates"][0]
    assert actual["sender_email"] == "recruiter@example.com"
    assert actual["sender_name"] == "Recruiter"
    assert actual["received_at"] == "2026-10-19T09:00:00+00:00"
    assert actual["body"] == candidate().page_content
    assert actual["subject"] == "Technical interview confirmation"


@pytest.mark.parametrize("response", [
    "Here are some random interview emails.",
    '{"decisions": []}',
    json.dumps({"decisions": [decision("invented-email")]}),
    json.dumps({"decisions": [{**decision(), "relevant": "true"}]}),
    json.dumps({"decisions": [{**decision(), "key_points": []}]}),
])
def test_invalid_llm_decisions_never_become_sources(response):
    result = validate_email_relevance([candidate()], "Find interviews", FakeListChatModel(responses=[response]))
    assert result.failed
    assert result.documents == []


def test_provider_failure_is_not_reported_as_no_matching_email():
    def fail(_):
        raise RuntimeError("Provider unavailable")
    result = validate_email_relevance([candidate()], "Find interviews", RunnableLambda(fail))
    assert result.failed
    assert result.documents == []


def test_multiple_validation_batches_keep_every_accepted_source():
    def respond(messages):
        data = json.loads(messages[-1].content)
        return AIMessage(content=json.dumps({"decisions": [decision(c["email_id"]) for c in data["candidates"]]}))
    result = validate_email_relevance([candidate(str(i)) for i in range(7)], "Find all interviews", RunnableLambda(respond))
    assert not result.failed
    assert len(result.documents) == result.evaluated == 7

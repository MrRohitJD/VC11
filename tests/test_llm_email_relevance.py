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


def test_bad_json_uses_next_provider_instead_of_failing_entire_search():
    bad = FakeListChatModel(responses=["not JSON"])
    good = FakeListChatModel(responses=[json.dumps({"decisions": [decision()]})])
    result = validate_email_relevance([candidate()], "Find interviews", bad.with_fallbacks([good]))
    assert not result.failed
    assert len(result.documents) == 1


def test_structured_message_content_blocks_are_parsed():
    model = RunnableLambda(lambda _: AIMessage(content=[{"type": "text", "text": json.dumps({"decisions": [decision()]})}]))
    assert not validate_email_relevance([candidate()], "Find interviews", model).failed


def test_last_three_months_uses_calendar_months_and_preserves_time():
    with mailbox_time_context("Asia/Kolkata"):
        start, end = received_date_bounds("Find all interviews from last 3 months", datetime(2026, 10, 19, 19, 24))
    assert start == datetime(2026, 7, 19, 13, 54, tzinfo=timezone.utc)
    assert end == datetime(2026, 10, 19, 13, 54, tzinfo=timezone.utc)


def test_month_subtraction_clamps_day_and_handles_year_boundary():
    with mailbox_time_context("Asia/Kolkata"):
        start, _ = received_date_bounds("past 3 months", datetime(2026, 5, 31))
    assert start == datetime(2026, 2, 27, 18, 30, tzinfo=timezone.utc)


def test_date_only_summary_does_not_misclassify_topic_or_sender_searches():
    from app.rag.search_plan import date_only_request, exhaustive_request
    assert date_only_request("Give me a brief summary of all incoming emails received today.")
    assert exhaustive_request("Give me a brief summary of all incoming emails received today.")
    assert not date_only_request("Summarize interview emails today")
    assert not date_only_request("Summarize emails from Alice today")


def test_malformed_batch_recovers_with_individual_strict_decisions():
    def respond(messages):
        data = json.loads(messages[-1].content)
        if len(data["candidates"]) > 1:
            return AIMessage(content="truncated JSON")
        return AIMessage(content=json.dumps({"decisions": [decision(data["candidates"][0]["email_id"])]}))
    result = validate_email_relevance([candidate("one"), candidate("two")], "Find all interviews", RunnableLambda(respond))
    assert not result.failed
    assert result.evaluated == 2
    assert len(result.documents) == 2


@pytest.mark.parametrize('words,digits', [('two weeks','2 weeks'), ('three months','3 months'), ('a year','1 year')])
def test_spelled_out_receipt_windows(words, digits):
    now = datetime(2026, 10, 19, tzinfo=timezone.utc)
    assert received_date_bounds('last ' + words, now) == received_date_bounds('last ' + digits, now)


def test_sender_constraints_do_not_confuse_dates_or_topic_mentions():
    from app.rag.search_plan import requested_sender, sender_only_request, sender_matches
    assert requested_sender('all emails from last two weeks') is None
    assert requested_sender('all emails from 2026-10-01') is None
    assert requested_sender('all emails from October') is None
    assert requested_sender('all emails from ChatGPT and summarize them') == 'ChatGPT'
    assert not sender_only_request('all emails from ChatGPT and summarize them')
    assert sender_only_request('all emails from ChatGPT in the last two weeks')
    assert sender_matches('ChatGPT', 'ChatGPT <noreply@example.com>')
    assert not sender_matches('ChatGPT', 'NotChatGPT <other@example.com>')
    assert sender_matches('alice@example.com', 'Alice <alice@example.com>')
    assert not sender_matches('alice@example.com', 'alice@example.com.evil.test')


def test_later_provider_failure_retains_verified_matches():
    def respond(messages):
        c = json.loads(messages[-1].content)['candidates'][0]
        if c['email_id'] == 'two':
            raise TimeoutError()
        return AIMessage(content=json.dumps({'decisions': [decision(c['email_id'])]}))
    result = validate_email_relevance([candidate('one'), candidate('two')], 'interviews', RunnableLambda(respond), batch_size=1)
    assert result.failed and result.error_type == 'timeout'
    assert result.evaluated == 1
    assert [d.metadata['email_id'] for d in result.documents] == ['one']


def test_bad_single_decision_does_not_discard_other_decisions():
    def respond(messages):
        cs = json.loads(messages[-1].content)['candidates']
        if len(cs) > 1 or cs[0]['email_id'] == 'bad':
            return AIMessage(content='invalid JSON')
        return AIMessage(content=json.dumps({'decisions': [decision(cs[0]['email_id'])]}))
    result = validate_email_relevance([candidate('one'), candidate('bad'), candidate('three')], 'interviews', RunnableLambda(respond))
    assert result.failed
    assert result.evaluated == 2
    assert [d.metadata['email_id'] for d in result.documents] == ['one', 'three']


def test_schema_failure_is_not_hidden_by_broken_fallback_credentials():
    def respond(messages):
        cs = json.loads(messages[-1].content)['candidates']
        return AIMessage(content='bad JSON' if len(cs) > 1 else json.dumps({'decisions': [decision(cs[0]['email_id'])]}))
    calls = []
    class AuthError(Exception):
        status_code = 401
    def broken(_):
        calls.append(1)
        raise AuthError()
    model = RunnableLambda(respond).with_fallbacks([RunnableLambda(broken)])
    result = validate_email_relevance([candidate('one'), candidate('two')], 'interviews', model)
    assert not result.failed
    assert len(result.documents) == 2
    assert len(calls) == 1

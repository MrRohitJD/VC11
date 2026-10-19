from datetime import datetime

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, AIMessage

from app.rag.agent import QUERY_TRANSFORM_PROMPT
from app.rag.query_constraints import billing_email_matches
from app.rag.retrieval import EmailKeyPointsCompressor, documents_to_citations, rewrite_query_with_history


@pytest.mark.parametrize("query", [
    "What are my pending bills and invoices due this month?",
    "Summarize emails received this week.",
    "What is due today?",
])
def test_standalone_question_does_not_inherit_previous_topic(query):
    history = [HumanMessage(content="Find interview related emails."), AIMessage(content="Your interview is Monday.")]
    llm = FakeListChatModel(responses=["Wrong interview rewrite", "Another response"])
    assert rewrite_query_with_history(query, history, llm=llm, prompt=QUERY_TRANSFORM_PROMPT) == query
    assert llm.i == 0


def test_true_followup_works_with_agent_prompt_variables():
    llm = FakeListChatModel(responses=["When is the technical interview scheduled?", "unused"])
    result = rewrite_query_with_history("When is it scheduled?", [HumanMessage(content="Find technical interviews.")],
                                        llm=llm, prompt=QUERY_TRANSFORM_PROMPT)
    assert result == "When is the technical interview scheduled?"
    assert llm.i == 1


def test_compression_respects_explicit_irrelevance():
    compressor = EmailKeyPointsCompressor(llm=FakeListChatModel(responses=["NO_RELEVANT_INFO"]))
    docs = [Document(page_content="Interview documents are pending.", metadata={"email_id": "one"})]
    assert compressor.compress_documents(docs, "What bills are due?") == []


def test_legacy_index_metadata_keeps_sender_and_date():
    citation = documents_to_citations([Document(page_content="Full invoice information.", metadata={
        "email_id": "one", "sender_email": "billing@example.com", "received_date": "2026-10-19T09:00:00+00:00",
    })])[0]
    assert citation.sender == "billing@example.com"
    assert citation.date == "2026-10-19T09:00:00+00:00"


@pytest.mark.parametrize("body,expected", [
    ("Your unpaid invoice amount due is INR 100. Payment due on 2026-10-25.", True),
    ("Your unpaid invoice amount due is INR 100. Due date: 25-Oct-2026.", True),
    ("Your unpaid invoice amount due is INR 100. Due October 25, 2026.", True),
    ("Your unpaid invoice amount due is INR 100. Due 25/10/2026.", True),
    ("Your unpaid invoice amount due is INR 100. Payment due on 2024-10-25.", False),
    ("Invoice paid in full. Payment due on 2026-10-25.", False),
    ("Interview documents are pending. Deadline: 2026-10-25.", False),
    ("Invoice available. The payment due date is not specified.", False),
    ("Invoice available. Due date is not specified. Invoice issued October 5, 2026.", False),
])
def test_billing_requires_payment_status_and_due_date_evidence(body, expected):
    assert billing_email_matches("What are my pending bills and invoices due this month?", "", body,
                                  datetime(2026, 9, 20), now=datetime(2026, 10, 19)) == expected


def test_yearless_historic_due_date_does_not_become_current():
    assert not billing_email_matches("Pending bills due this month", "Invoice",
                                      "Amount due INR 100. Payment due on October 25.",
                                      datetime(2024, 10, 20), now=datetime(2026, 10, 19))


def test_plain_invoice_query_does_not_require_a_due_date():
    assert billing_email_matches("Show invoices received this month", "Invoice available", "Your invoice is ready.",
                                 datetime(2026, 10, 10), now=datetime(2026, 10, 19))


def test_offline_compression_drops_unrelated_results():
    compressor = EmailKeyPointsCompressor()
    docs = [Document(page_content="Your interview documents are pending.", metadata={"subject": "Pending documents"})]
    assert compressor.compress_documents(docs, "What bills and invoices are pending this month?") == []


def test_timezone_context_is_restored_after_request():
    from app.rag.query_constraints import mailbox_now, mailbox_time_context
    original = str(mailbox_now().tzinfo)
    with mailbox_time_context("UTC"):
        assert str(mailbox_now().tzinfo) == "UTC"
    assert str(mailbox_now().tzinfo) == original

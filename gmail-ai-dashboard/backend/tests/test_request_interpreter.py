"""Approval-path regressions using real graph/tools and mocked provider responses."""
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError
from langgraph.checkpoint.memory import InMemorySaver

from app.rag.agent import GmailAIAssistant, deterministic_action_answer
from app.rag.graph_agent import build_mailbox_graph
from app.rag.request_interpreter import EmailDraft, InterpretedRequest, interpret_request


QUERIES = [
    "Send a mail to the class teacher related to leave approval for next week's Monday and Tuesday. at mail address xyz@gmail.com",
    "Send a mail to my class teacher named Irshad, related to leave approval for next Friday and Saturday. The mail ID is xyz@adypu.edu.in.",
    "Could you put together a note asking my teacher for two days off? Use xyz@gmail.com.",
]


class FakeDB:
    def __init__(self, fail=False):
        self.records = []
        self.fail = fail

    def get(self, *args):
        return SimpleNamespace(is_demo=False)

    def add(self, record):
        self.records.append(record)

    def commit(self):
        if self.fail:
            from sqlalchemy.exc import SQLAlchemyError
            raise SQLAlchemyError("Cannot save")
        for record in self.records:
            record.id = uuid4()

    def rollback(self):
        self.records.clear()


class Provider:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def with_structured_output(self, schema):
        assert schema is InterpretedRequest
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def run_graph(query, result, fail_save=False, demo=False):
    db = FakeDB(fail_save)
    if demo:
        db.get = lambda *args: SimpleNamespace(is_demo=True)
    provider = Provider(result)
    assistant = GmailAIAssistant.__new__(GmailAIAssistant)
    assistant.model = provider
    assistant.max_agent_iterations = 3
    graph = build_mailbox_graph(assistant, db=db, user_id=str(uuid4()), history=[],
        top_k=5, filter_dict=None, use_multi_query=False, citations=[], checkpointer=InMemorySaver())
    state = graph.invoke({"query": query, "messages": [], "steps": [], "iterations": 0},
                         {"configurable": {"thread_id": str(uuid4())}})
    return state, db, provider


def composition(recipient="xyz@gmail.com"):
    return {"intent": "ACTION_DRAFT", "draft": {
        "recipient": recipient, "subject": "Request for Leave Approval",
        "body": "Dear Irshad,\n\nPlease approve my leave for Friday and Saturday.\n\nThank you.",
    }}


@pytest.mark.parametrize("query", QUERIES)
def test_composition_creates_actual_pending_approval_in_one_model_call(query):
    recipient = "xyz@adypu.edu.in" if "adypu.edu.in" in query else "xyz@gmail.com"
    result, db, provider = run_graph(query, composition(recipient))
    assert result["intent"] == "ACTION_DRAFT"
    assert len(db.records) == 1
    approval = db.records[0]
    assert approval.status == "PENDING"
    assert approval.recipient_email == recipient
    assert result["approval_id"] == str(approval.id)
    assert approval.draft_body == composition(recipient)["draft"]["body"]
    assert result["steps"][0]["action"] == "draft_email"
    assert len(provider.calls) == 1
    assert provider.calls[0][-1].content == query


@pytest.mark.parametrize("response", [
    TimeoutError("Provider timed out"),
    {"intent": "ACTION_DRAFT", "draft": None},
    {"intent": "nonexistent_tool"},
])
def test_failed_interpretation_creates_reviewable_leave_draft(response, monkeypatch):
    monkeypatch.setattr("app.rag.request_interpreter.mailbox_now", lambda: datetime(2026, 10, 28, 14, 36,
                                                                                  tzinfo=timezone(timedelta(hours=5, minutes=30))))
    result, db, provider = run_graph(QUERIES[0], response)
    assert result["intent"] == "ACTION_DRAFT"
    assert len(db.records) == 1
    assert db.records[0].status == "PENDING"
    assert db.records[0].recipient_email == "xyz@gmail.com"
    assert "Monday, November 2, 2026" in db.records[0].draft_body
    assert "Tuesday, November 3, 2026" in db.records[0].draft_body
    assert result["approval_id"] == str(db.records[0].id)
    assert "sent" not in result["answer"].lower()
    assert len(provider.calls) == (1 if isinstance(response, Exception) else 2)


def test_explicit_clarification_does_not_create_draft():
    result, db, _ = run_graph(QUERIES[0],
        {"intent": "CLARIFY", "clarification_question": "Which teacher should receive it?"})
    assert result["intent"] == "CLARIFY"
    assert result["answer"] == "Which teacher should receive it?"
    assert not db.records


def test_provider_failure_without_explicit_recipient_does_not_create_draft():
    result, db, _ = run_graph("Send a mail to my class teacher about leave next Monday and Tuesday",
                              TimeoutError("Provider timed out"))
    assert result["intent"] == "CLARIFY"
    assert not db.records
    assert not result.get("approval_id")


def test_save_failure_does_not_claim_approval_exists():
    result, db, _ = run_graph(QUERIES[0], composition(), fail_save=True)
    assert not result["approval_id"]
    assert "created a professional draft" not in result["answer"]


def test_demo_mailbox_does_not_create_outbound_approval():
    result, db, _ = run_graph(QUERIES[0], composition(), demo=True)
    assert not db.records
    assert not result.get("approval_id")
    assert "unavailable" in result["answer"]


def test_interpreter_uses_history_and_current_datetime():
    from langchain_core.messages import HumanMessage
    provider = Provider({"intent": "RAG_QUERY"})
    history = [HumanMessage(content="My teacher is Irshad")]
    request = interpret_request(provider, "Find his latest email", history)
    assert request.intent == "RAG_QUERY"
    assert provider.calls[0][1] == history[0]
    assert "Current datetime:" in provider.calls[0][0].content


def test_draft_cannot_be_mistaken_for_send_authorization():
    with pytest.raises(ValidationError):
        InterpretedRequest(intent="ACTION_SEND", draft=EmailDraft(**composition()["draft"]))


def test_failed_tool_result_is_not_reported_as_success():
    failure = "Draft could not be created: Invalid recipient"
    assert deterministic_action_answer("draft_email", {}, failure) == failure


def test_structured_interpretation_falls_back_after_provider_timeout():
    from langchain_core.runnables import RunnableLambda
    calls = []

    class FailingProvider:
        def with_structured_output(self, schema):
            def invoke(messages):
                calls.append("primary")
                raise TimeoutError("Unavailable")
            return RunnableLambda(invoke)

    class WorkingProvider:
        def with_structured_output(self, schema):
            def invoke(messages):
                calls.append("fallback")
                return schema.model_validate(composition())
            return RunnableLambda(invoke)

    model = SimpleNamespace(runnable=FailingProvider(), fallbacks=[WorkingProvider()])
    request = interpret_request(model, QUERIES[0])
    assert request.intent == "ACTION_DRAFT"
    assert calls == ["primary", "fallback"]


def test_structured_interpretation_skips_provider_that_cannot_bind():
    class FailingBindingProvider:
        def with_structured_output(self, schema):
            raise ConnectionError("Provider discovery unavailable")

    working = Provider(composition())
    model = SimpleNamespace(runnable=FailingBindingProvider(), fallbacks=[working])
    request = interpret_request(model, QUERIES[0])
    assert request.intent == "ACTION_DRAFT"
    assert len(working.calls) == 1


@pytest.mark.parametrize("failure", [TimeoutError("offline"), {"intent": "unknown"}, None])
def test_summary_interpretation_failure_still_reaches_grounded_retrieval(failure, monkeypatch):
    from app.rag import graph_agent
    monkeypatch.setattr(graph_agent, "create_gmail_tools", lambda *_: [])
    query = "Summarize experimental lab mails in five bullet points"
    calls = []
    def answer_query(**kwargs):
        calls.append(kwargs)
        return {"answer": "Verified summary from retrieved mail", "citations": []}
    assistant = GmailAIAssistant.__new__(GmailAIAssistant)
    assistant.model = Provider(failure)
    assistant.rag_chain = SimpleNamespace(answer_query=answer_query)
    assistant.max_agent_iterations = 3
    graph = build_mailbox_graph(assistant, db=None, user_id=str(uuid4()), history=[],
        top_k=5, filter_dict=None, use_multi_query=False, citations=[], checkpointer=InMemorySaver())
    state = graph.invoke({"query": query, "messages": [], "steps": [], "iterations": 0},
                         {"configurable": {"thread_id": str(uuid4())}})
    assert state["intent"] == "SUMMARIZE"
    assert state["answer"] == "Verified summary from retrieved mail"
    assert len(calls) == 1
    assert calls[0]["query"] == query
    assert not state.get("approval_id")


def test_analysis_classification_avoids_interpreter_timeout_and_reaches_mailbox_read(monkeypatch):
    from app.rag import graph_agent
    monkeypatch.setattr(graph_agent, "create_gmail_tools", lambda *_: [])
    query = "Analyze my recent emails and categorize them as Work, Job Opportunities, Education, Promotions, Social, or Important. Explain each classification briefly."
    calls = []
    assistant = GmailAIAssistant.__new__(GmailAIAssistant)
    assistant.model = Provider(TimeoutError("Provider timed out"))
    assistant.rag_chain = SimpleNamespace(answer_query=lambda **kwargs: (
        calls.append(kwargs) or {"answer": "Classified mailbox emails", "citations": []}))
    assistant.max_agent_iterations = 3
    graph = build_mailbox_graph(assistant, db=None, user_id=str(uuid4()), history=[],
        top_k=5, filter_dict=None, use_multi_query=False, citations=[], checkpointer=InMemorySaver())
    state = graph.invoke({"query": query, "messages": [], "steps": [], "iterations": 0},
                         {"configurable": {"thread_id": str(uuid4())}})
    assert state["intent"] == "RAG_QUERY"
    assert state["answer"] == "Classified mailbox emails"
    assert calls[0]["query"] == query
    assert assistant.model.calls == []


@pytest.mark.parametrize("query", [
    "Summarize lab mails and send the result to xyz@gmail.com",
    "Find lab emails then delete them",
    "Summarize those emails in five points",
    "Summarize his emails",
    "Summarize the attached document",
])
def test_read_fallback_declines_actions_and_unresolved_context(query):
    from app.rag.request_interpreter import fallback_read_request
    assert fallback_read_request(query) is None


@pytest.mark.parametrize("bad_response", [None, {"intent": "unknown"}, {"intent": "ACTION_DRAFT"}])
def test_invalid_primary_response_tries_next_provider(bad_response):
    primary = Provider(bad_response)
    fallback = Provider({"intent": "SUMMARIZE"})
    model = SimpleNamespace(runnable=primary, fallbacks=[fallback])
    result = interpret_request(model, "Summarize experimental lab mails in five bullet points")
    assert result.intent == "SUMMARIZE"
    assert len(primary.calls) == len(fallback.calls) == 1


def test_interpretation_error_distinguishes_validation_from_outage():
    from app.rag.request_interpreter import interpretation_failure_message
    assert "could not be validated" in interpretation_failure_message(ValidationError.from_exception_data("Request", []))
    assert "time limit" in interpretation_failure_message(TimeoutError())
    assert "Could not connect" in interpretation_failure_message(ConnectionError())


def test_programming_errors_are_not_reported_as_invalid_ai_output():
    from app.rag.service_errors import error_code
    for error in (ValueError("bad config"), KeyError("secret"), TypeError("bad call")):
        assert error_code(error) == "internal_error"


def test_invalid_response_is_repaired_once_with_private_diagnostics(caplog):
    class RepairProvider(Provider):
        def invoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return {"intent": "ACTION_DRAFT", "draft": {
                    "recipient": "private@example.com", "subject": "private subject", "body": ""}}
            return composition()
    provider = RepairProvider(None)
    result = interpret_request(provider, "Please prepare my message")
    assert result.intent == "ACTION_DRAFT"
    assert len(provider.calls) == 2
    assert "draft.body" in provider.calls[-1][0].content
    assert "private@example.com" not in caplog.text
    assert "private subject" not in caplog.text
    assert "string_too_short" in caplog.text


def test_invalid_output_repair_is_bounded_and_never_grants_send_permission():
    provider = Provider({"intent": "ACTION_SEND", "draft": composition()["draft"]})
    with pytest.raises(ValidationError):
        interpret_request(provider, "Prepare a message for my teacher")
    assert len(provider.calls) == 2


@pytest.mark.parametrize("error", [TimeoutError(), TypeError("bug"), ConnectionError()])
def test_non_output_errors_are_not_retried(error):
    provider = Provider(error)
    with pytest.raises(type(error)):
        interpret_request(provider, "Hello")
    assert len(provider.calls) == 1


def test_openai_compatible_json_content_is_validated():
    import json
    from langchain_core.messages import AIMessage
    from app.rag.request_interpreter import _validated_interpretation
    response = {"raw": AIMessage(content=json.dumps(composition())), "parsed": None}
    assert _validated_interpretation(response).intent == "ACTION_DRAFT"
    response["raw"] = AIMessage(content='{"intent":"ACTION_DRAFT"}')
    with pytest.raises(ValidationError):
        _validated_interpretation(response)


@pytest.mark.parametrize("metadata,kwargs", [
    ({"finish_reason": "length"}, {}),
    ({"finish_reason": "content_filter"}, {}),
    ({}, {"refusal": "declined"}),
])
def test_incomplete_or_refused_outputs_are_not_used(metadata, kwargs):
    from langchain_core.messages import AIMessage
    from app.rag.request_interpreter import _validated_interpretation, InterpretationResponseError
    with pytest.raises(InterpretationResponseError):
        _validated_interpretation({"raw": AIMessage(content="", response_metadata=metadata,
                                                   additional_kwargs=kwargs),
                                   "parsed": InterpretedRequest(intent="ACTION_SEND")})


def test_azure_legacy_api_uses_tool_calling_without_json_schema():
    import httpx
    import json
    from langchain_openai import AzureChatOpenAI
    payloads = []
    def completion(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        # Reproduce an endpoint that rejects the newer response-format API.
        if "response_format" in payload:
            return httpx.Response(400, json={"error": {"message": "response_format unsupported",
                                                       "type": "invalid_request_error"}})
        return httpx.Response(200, json={"id": "test", "object": "chat.completion",
            "created": 1, "model": "gpt-4.1-mini", "choices": [{"index": 0,
            "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "call_test", "type": "function", "function": {
                "name": "InterpretedRequest", "arguments": json.dumps(composition())}}]}}]})
    with httpx.Client(transport=httpx.MockTransport(completion)) as client:
        model = AzureChatOpenAI(azure_endpoint="https://test.openai.azure.com",
            azure_deployment="test", api_version="2024-02-01", api_key="test-only",
            http_client=client, max_retries=0)
        # The previous implementation fails against the same endpoint.
        from openai import BadRequestError
        with pytest.raises(BadRequestError):
            model.with_structured_output(InterpretedRequest).invoke("Prepare my message")
        assert interpret_request(model, "Prepare my message").intent == "ACTION_DRAFT"
    assert "response_format" not in payloads[-1]
    assert payloads[-1]["tools"][0]["function"]["name"] == "InterpretedRequest"


def test_concurrent_tabs_keep_query_and_history_separate():
    from concurrent.futures import ThreadPoolExecutor
    from langchain_core.messages import HumanMessage
    class ConcurrentProvider:
        def with_structured_output(self, schema):
            return self
        def invoke(self, messages):
            return {"intent": "GENERAL_CHAT", "effective_query":
                    messages[-2].content + ":" + messages[-1].content}
    provider = ConcurrentProvider()
    def ask(index):
        return interpret_request(provider, f"question-{index}",
                                 [HumanMessage(content=f"tab-{index}")]).effective_query
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(ask, range(10)))
    assert results == [f"tab-{i}:question-{i}" for i in range(10)]

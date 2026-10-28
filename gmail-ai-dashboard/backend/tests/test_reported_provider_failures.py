"""Regressions for reported financial-search and teacher-leave failures."""
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import app.models.llm
from app.models.email import Base, Email, GmailAccount, User
from app.rag.query_constraints import billing_email_matches
from app.rag.request_interpreter import interpret_request, interpretation_failure_message
from app.rag.search_plan import gmail_search_query
from app.rag.service_errors import error_code
from app.services import gmail_mcp_server as server


LEAVE = "Send a mail to a class teacher related to approving my leave application for Diwali vacation. I need 15 days of leave during Diwali dates."
FINANCIAL = "Find emails about bills, payments, and receipts from the last 30 days. Show the amount, payment date, and whether anything still needs to be paid."


@pytest.mark.parametrize("address", ["", ' mail id " zyx@gmail.com"'])
def test_holiday_leave_asks_for_missing_facts_without_an_ai_call(address):
    request = interpret_request(None, LEAVE.replace("class teacher", "class teacher" + address))
    assert request.intent == "CLARIFY"
    assert request.draft is None
    assert "15-day" in request.clarification_question
    assert "start date (including year)" in request.clarification_question
    assert ("teacher's email address" in request.clarification_question) == (not address)


def test_explicit_teacher_leave_gets_clarification_after_provider_outage_with_history():
    from langchain_core.messages import HumanMessage
    class Provider:
        def with_structured_output(self, schema):
            return self
        def invoke(self, messages):
            raise TimeoutError()
    request = interpret_request(Provider(), LEAVE + " Teacher email zyx@gmail.com.",
                                [HumanMessage(content="Hello")])
    assert request.intent == "CLARIFY"
    assert "start date" in request.clarification_question


def test_mixed_provider_failure_reports_both_providers_without_secret_text():
    class AzureChatOpenAI:
        def with_structured_output(self, schema):
            return self
        def invoke(self, messages):
            raise TimeoutError("private prompt")
    class ChatNVIDIA(AzureChatOpenAI):
        def invoke(self, messages):
            error = RuntimeError("secret API key")
            error.status_code = 401
            raise error
    model = SimpleNamespace(runnable=AzureChatOpenAI(), fallbacks=[ChatNVIDIA()])
    with pytest.raises(RuntimeError) as failure:
        interpret_request(model, "Hello")
    message = interpretation_failure_message(failure.value)
    assert "Azure OpenAI: timeout" in message
    assert "NVIDIA: authentication" in message
    assert "secret" not in message and "private prompt" not in message


def test_financial_discovery_keeps_dates_and_targets_receipts_as_well_as_bills():
    query = gmail_search_query(FINANCIAL)
    assert "after:" in query and "before:" in query
    assert "receipt receipts" in query and "invoice invoices" in query
    assert "-in:spam" in query
    assert "receipt receipts" not in gmail_search_query("Find interview emails from the last 30 days")
    assert billing_email_matches(FINANCIAL, "Your receipt", "INR 499 purchase confirmed.",
                                 datetime.now(timezone.utc))


def test_invalid_relevance_output_is_not_misdiagnosed_as_an_application_bug():
    from app.rag.relevance import _parse_decisions
    for content in ('not JSON', '{}', '{"decisions": []}'):
        with pytest.raises(ValueError) as failure:
            _parse_decisions(SimpleNamespace(content=content), [{"email_id": "one"}])
        assert error_code(failure.value) == "invalid_response"


@pytest.fixture
def mailbox():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(google_id=str(uuid4()), email="owner@example.com")
        db.add(user)
        db.flush()
        db.add(GmailAccount(user_id=user.id, email=user.email, access_token="test-only"))
        db.commit()
        yield db, user
    engine.dispose()


class Gmail:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.timeouts = []
        self._http = SimpleNamespace(http=SimpleNamespace(timeout=None))
    def users(self):
        return self
    def messages(self):
        return self
    def list(self, **kwargs):
        self.calls.append(kwargs)
        self.timeouts.append(self._http.http.timeout)
        def execute():
            self.clock[0] += 1
            if kwargs.get("pageToken"):
                raise TimeoutError()
            return {"messages": [{"id": "receipt-one"}], "nextPageToken": "next"}
        return SimpleNamespace(execute=execute)
    def get(self, **kwargs):
        return SimpleNamespace(execute=lambda: {"id": kwargs["id"], "internalDate": "1792396800000",
            "labelIds": ["INBOX", "CATEGORY_UPDATES"], "payload": {"headers": [
                {"name": "Subject", "value": "Payment receipt"}]}})
    def new_batch_http_request(self, callback):
        requests = []
        def execute():
            self.timeouts.append(self._http.http.timeout)
            self.clock[0] += 1
            for request_id, request in requests:
                callback(request_id, request.execute(), None)
        return SimpleNamespace(add=lambda request, request_id: requests.append((request_id, request)),
                               execute=execute)


def test_live_search_bounds_network_operations_and_preserves_completed_pages(mailbox, monkeypatch):
    db, owner = mailbox
    clock = [100.0]
    gmail = Gmail(clock)
    monkeypatch.setattr(server, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    builders = []
    def build(db, user, *, timeout_seconds):
        builders.append(timeout_seconds)
        return gmail
    monkeypatch.setattr(server, "build_gmail_service_for_user", build)
    result = server.search_mailbox(db, owner.id, "receipt", 100, time_budget_seconds=5)
    assert builders == [5]
    assert result["error_code"] == "timeout" and not result["complete"]
    assert len(result["email_ids"]) == 1
    assert db.scalar(select(Email)).gmail_account.user_id == owner.id
    assert gmail.calls[0]["maxResults"] == 20
    assert gmail.timeouts == [5, 4, 3]


def test_exhausted_budget_never_starts_gmail_network_calls(mailbox, monkeypatch):
    db, owner = mailbox
    monkeypatch.setattr(server, "build_gmail_service_for_user", lambda *args, **kwargs: pytest.fail("No budget"))
    result = server.search_mailbox(db, owner.id, "receipt", 100, time_budget_seconds=0)
    assert result["error_code"] == "timeout"
    assert result["email_ids"] == []


def test_gmail_client_uses_bounded_transport_without_discovery_network_call(mailbox, monkeypatch):
    from app.services import gmail_service
    db, owner = mailbox
    calls = []
    monkeypatch.setattr(gmail_service, "build", lambda *args, **kwargs: calls.append(kwargs) or kwargs)
    result = gmail_service.build_gmail_service_for_user(db, owner, timeout_seconds=4)
    assert result["http"].http.timeout == 4
    assert result["static_discovery"] is True
    assert result["cache_discovery"] is False


def test_gmail_token_refresh_uses_the_live_search_timeout(mailbox, monkeypatch):
    from datetime import timedelta
    from app.services import gmail_service
    db, owner = mailbox
    account = db.scalar(select(GmailAccount))
    account.token_expiry = datetime.now(timezone.utc) - timedelta(hours=1)
    account.refresh_token = "test-refresh-only"
    db.commit()
    timeouts = []
    class Credentials:
        expired = True
        token = "refreshed-test-only"
        refresh_token = "test-refresh-only"
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        def __init__(self, **kwargs):
            pass
        def refresh(self, request):
            request("https://test.invalid/token")
    monkeypatch.setattr(gmail_service, "Credentials", Credentials)
    monkeypatch.setattr(gmail_service, "GoogleAuthRequest", lambda: (
        lambda url, **kwargs: timeouts.append(kwargs["timeout"])))
    monkeypatch.setattr(gmail_service, "build", lambda *args, **kwargs: kwargs)
    gmail_service.build_gmail_service_for_user(db, owner, timeout_seconds=3)
    assert timeouts == [3]
    assert account.access_token == "refreshed-test-only"


def test_mcp_passes_remaining_budget_after_startup_and_handshake(monkeypatch):
    from contextlib import asynccontextmanager
    import mcp
    import mcp.client.stdio
    from app.rag import gmail_mcp
    calls = []
    clock = [100.0]
    @asynccontextmanager
    async def stdio_client(params):
        clock[0] += 4
        yield None, None
    class ClientSession:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def initialize(self):
            clock[0] += 1
        async def call_tool(self, name, arguments):
            calls.append(arguments)
            return SimpleNamespace(isError=False, structuredContent={"email_ids": [], "complete": False}, content=[])
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", stdio_client)
    monkeypatch.setattr(mcp, "ClientSession", ClientSession)
    monkeypatch.setattr(gmail_mcp, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(gmail_mcp.settings, "gmail_mcp_timeout_seconds", 15)
    @asynccontextmanager
    async def timeout(seconds):
        yield
    monkeypatch.setattr(gmail_mcp, "asyncio", SimpleNamespace(timeout=timeout))
    # All async I/O is mocked and completes synchronously; drive this coroutine
    # without an OS event loop or Windows loopback socket requirements.
    coroutine = gmail_mcp._search("owner", FINANCIAL)
    with pytest.raises(StopIteration):
        coroutine.send(None)
    assert calls[0]["time_budget_seconds"] == 8

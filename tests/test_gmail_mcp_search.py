"""MCP discovery pagination, read-only behavior and account isolation."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import app.models.llm
from app.models.email import Base, User, GmailAccount, Email
from app.services import gmail_mcp_server as server
from app.rag.search_plan import gmail_search_query


@pytest.fixture
def mailbox():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        users = [User(google_id=str(i), email=f"u{i}@example.com") for i in range(2)]
        db.add_all(users)
        db.flush()
        for user in users:
            db.add(GmailAccount(user_id=user.id, email=user.email, access_token="test"))
        db.commit()
        yield db, users
    engine.dispose()


class Gmail:
    def __init__(self):
        self.calls = []

    def users(self):
        return self

    def messages(self):
        return self

    def new_batch_http_request(self, callback):
        requests = []
        def execute():
            for request_id, request in requests:
                callback(request_id, request.execute(), None)
        return SimpleNamespace(add=lambda request, request_id: requests.append((request_id, request)), execute=execute)

    def list(self, **kwargs):
        self.calls.append(kwargs)
        result = {"messages": [{"id": "second"}]} if kwargs.get("pageToken") else {
            "messages": [{"id": "first"}], "nextPageToken": "page2"}
        return SimpleNamespace(execute=lambda: result)

    def get(self, **kwargs):
        return SimpleNamespace(execute=lambda: {
            "id": kwargs["id"], "internalDate": "1792396800000",
            "labelIds": ["INBOX", "CATEGORY_PERSONAL"],
            "payload": {"headers": [{"name": "Subject", "value": "Interview"}]},
        })


def test_live_search_pages_and_never_crosses_accounts(mailbox, monkeypatch):
    db, users = mailbox
    gmail = Gmail()
    monkeypatch.setattr(server, "build_gmail_service_for_user", lambda db, user: gmail)
    result = server.search_primary(db, users[0].id, gmail_search_query("received today"), 10)
    assert result["complete"]
    assert len(result["email_ids"]) == 2
    assert gmail.calls[1]["pageToken"] == "page2"
    emails = list(db.scalars(select(Email)))
    assert all(e.gmail_account.user_id == users[0].id for e in emails)
    assert not any(e.gmail_account.user_id == users[1].id for e in emails)
    assert all(e.received_at is not None for e in emails)


def test_limit_is_reported_as_incomplete(mailbox, monkeypatch):
    db, users = mailbox
    monkeypatch.setattr(server, "build_gmail_service_for_user", lambda db, user: Gmail())
    result = server.search_primary(db, users[0].id, "category:primary", 1)
    assert not result["complete"]
    assert len(result["email_ids"]) == 1


def test_server_rechecks_labels_even_if_query_is_broad(mailbox, monkeypatch):
    db, users = mailbox
    gmail = Gmail()
    original_get = gmail.get
    def get(**kwargs):
        response = original_get(**kwargs).execute()
        response["labelIds"] = ["INBOX", "CATEGORY_UPDATES"]
        return SimpleNamespace(execute=lambda: response)
    monkeypatch.setattr(gmail, "get", get)
    monkeypatch.setattr(server, "build_gmail_service_for_user", lambda db, user: gmail)
    result = server.search_primary(db, users[0].id, "", 10)
    assert result["email_ids"] == []


def test_query_keeps_primary_scope_and_has_epoch_boundaries():
    query = gmail_search_query("Find all interview emails from last 3 months")
    assert "-category:updates" in query
    assert "after:" in query and "before:" in query
    assert "{category:primary in:inbox}" in query
    assert "newer_than:1y" not in query


def test_sender_query_is_narrowed_before_live_fetch():
    assert 'from:"ChatGPT"' in gmail_search_query('Give all mails from ChatGPT.')
    query = gmail_search_query('Find all interviews from last two weeks')
    assert 'after:' in query and 'from:"last' not in query


def test_mcp_import_does_not_construct_ai_or_vector_clients():
    import subprocess
    import sys
    from pathlib import Path
    result = subprocess.run([sys.executable, '-c',
        "import sys; import app.services.gmail_mcp_server; assert 'app.rag.chains' not in sys.modules; assert 'app.rag.vectorstore' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1] / 'gmail-ai-dashboard' / 'backend', capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors='replace')


def test_later_page_failure_preserves_committed_results(mailbox, monkeypatch):
    db, users = mailbox
    gmail = Gmail()
    original = gmail.list
    def failing_list(**kwargs):
        if kwargs.get('pageToken'):
            raise TimeoutError()
        return original(**kwargs)
    gmail.list = failing_list
    monkeypatch.setattr(server, 'build_gmail_service_for_user', lambda *a: gmail)
    result = server.search_primary(db, users[0].id, 'category:primary', 10)
    assert not result['complete']
    assert result['error_code'] == 'timeout'
    assert len(result['email_ids']) == 1
    assert len(list(db.scalars(select(Email)))) == 1


def test_excluded_messages_still_count_towards_fetch_limit(mailbox, monkeypatch):
    db, users = mailbox
    gmail = Gmail()
    original = gmail.get
    def excluded(**kwargs):
        message = original(**kwargs).execute()
        message['labelIds'] = ['CATEGORY_UPDATES']
        return SimpleNamespace(execute=lambda: message)
    gmail.get = excluded
    monkeypatch.setattr(server, 'build_gmail_service_for_user', lambda *a: gmail)
    result = server.search_primary(db, users[0].id, '', 1)
    assert not result['complete']
    assert result['email_ids'] == []
    assert len(gmail.calls) == 1


def test_stdio_mcp_server_exposes_only_read_only_search():
    import asyncio
    import os
    import sys
    from pathlib import Path
    from uuid import uuid4
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def check():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "app.services.gmail_mcp_server"],
            cwd=str(Path(__file__).resolve().parents[1] / "gmail-ai-dashboard" / "backend"),
            env={**os.environ, "GMAIL_MCP_USER_ID": str(uuid4())},
        )
        async with asyncio.timeout(30):
            async with stdio_client(params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    assert [tool.name for tool in result.tools] == ["search_primary_emails"]
                    assert "user_id" not in result.tools[0].inputSchema.get("properties", {})
    asyncio.run(check())

"""Test Gmail MCP integration in the core of RAG."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import app.models.llm
from app.models.email import Base, Email, GmailAccount, User
from app.rag import chains, gmail_mcp
from app.rag.chains import GroundedRAGChain
from app.rag.relevance import RelevanceResult
from app.rag.search_plan import extract_gmail_search_terms, gmail_search_query
from app.rag.tools import create_gmail_tools
from app.rag.graph_agent import READ_ONLY_TOOL_NAMES
from app.services import gmail_mcp_server as server


def test_extract_gmail_search_terms_from_user_query():
    # Regular topic search
    terms = extract_gmail_search_terms("Show me emails about python workshop and async programming")
    assert "python" in terms
    assert "workshop" in terms
    assert "async" in terms
    assert "programming" in terms
    assert "show" not in terms
    assert "emails" not in terms

    # Quoted phrase
    quoted_terms = extract_gmail_search_terms('Find messages regarding "Project Apollo" updates')
    assert '"Project Apollo"' in quoted_terms

    # Date-only query returns no topic terms
    date_terms = extract_gmail_search_terms("Show all emails from last week")
    assert date_terms == []

    # Sender-only query returns no topic terms
    sender_terms = extract_gmail_search_terms("Show emails from alex@example.com")
    assert sender_terms == []


def test_gmail_search_query_builds_targeted_query_based_on_user_query():
    query = gmail_search_query("Find emails about flight booking to Paris from the last 10 days")
    assert "-in:spam" in query
    assert "-in:trash" in query
    assert "-in:drafts" in query
    assert "after:" in query and "before:" in query
    assert "flight" in query
    assert "booking" in query
    assert "Paris" in query
    broad = gmail_search_query("Find all flight booking emails from the last 10 days",
                               include_topic_terms=False)
    assert "after:" in broad and "before:" in broad
    assert "flight" not in broad and "booking" not in broad


def test_rag_core_uses_gmail_mcp_and_keeps_both_local_and_live_mails(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(google_id="test_user", email="test@example.com")
        db.add(user)
        db.flush()
        account = GmailAccount(user_id=user.id, email=user.email, access_token="mock_token")
        db.add(account)
        db.flush()

        # 1. Existing local email in database (synced)
        local_email = Email(
            gmail_account_id=account.id,
            gmail_message_id="msg_local_1",
            from_email="alice@company.com",
            subject="Local Project Meeting",
            body="Meeting notes for local project discussion.",
            snippet="Meeting notes for local project discussion.",
            received_at=datetime.now(timezone.utc),
        )
        # 2. Fresh email discovered by Gmail MCP
        mcp_email = Email(
            gmail_account_id=account.id,
            gmail_message_id="msg_mcp_2",
            from_email="bob@company.com",
            subject="MCP Live Project Update",
            body="Live update from Gmail MCP with fresh project details.",
            snippet="Live update from Gmail MCP with fresh project details.",
            received_at=datetime.now(timezone.utc),
        )
        db.add_all([local_email, mcp_email])
        db.commit()

        mcp_id_str = str(mcp_email.id)
        # Mock MCP to return the newly discovered email ID.
        live_calls = []
        def live_search(user_id, query, full_scan=False):
            live_calls.append(full_scan)
            return {
                "email_ids": [mcp_id_str],
                "complete": True,
                "source": "gmail_mcp",
                "error_code": "",
            }
        monkeypatch.setattr(
            gmail_mcp,
            "search_live_mailbox",
            live_search,
        )

        # Mock semantic retriever for local emails in database
        monkeypatch.setattr(
            chains.PostgresSemanticRetriever,
            "invoke",
            lambda self, query: [
                Document(
                    page_content=local_email.body,
                    metadata={
                        "email_id": str(local_email.id),
                        "subject": local_email.subject,
                        "sender": local_email.from_email,
                        "thread_id": "",
                    },
                )
            ],
        )

        # Relevance validator allows both local and MCP emails
        def validate(documents, query, model, **kwargs):
            return RelevanceResult(documents=documents, evaluated=len(documents))

        monkeypatch.setattr(chains, "validate_email_relevance", validate)
        monkeypatch.setattr(
            chains,
            "generate_grounded_answer",
            lambda docs, query, model, **kwargs: "Synthesized grounded answer citing all sources.",
        )

        chain = GroundedRAGChain(model=FakeListChatModel(responses=["unused"]))
        
        # Broad queries automatically combine the local index and live Gmail.
        result = chain.answer_query(
            db=db,
            query="Find all project emails",
            user_id=user.id,
        )

        assert "gmail_mcp" in result["search_plan"]["providers"]
        assert live_calls == [True]
        assert result["context_count"] == 2
        cited_ids = {c.email_id for c in result["citations"]}
        # Both MCP-returned mail and local mail are preserved in context ("keep both")
        assert str(mcp_email.id) in cited_ids
        assert str(local_email.id) in cited_ids

        disabled = chain.answer_query(
            db=db, query="Find all project emails", user_id=user.id, use_gmail_mcp=False,
        )
        assert "gmail_mcp" not in disabled["search_plan"]["providers"]
        assert live_calls == [True]
    engine.dispose()


def test_gmail_mcp_tools_registered_and_read_only():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(google_id="tool_user", email="user@example.com", is_demo=False)
        db.add(user)
        db.commit()

        tools = create_gmail_tools(db, user.id)
        tool_names = {t.name for t in tools}

        assert "search_gmail_mcp" in tool_names
        assert "search_gmail_live" in tool_names
        assert "search_gmail_mcp" in READ_ONLY_TOOL_NAMES
        assert "search_gmail_live" in READ_ONLY_TOOL_NAMES
    engine.dispose()


def test_mcp_and_direct_gmail_tools_return_cited_mail(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(google_id="live_tool_user", email="live@example.com", is_demo=False)
        db.add(user)
        db.flush()
        account = GmailAccount(user_id=user.id, email=user.email, access_token="mock_token")
        db.add(account)
        db.flush()
        email = Email(
            gmail_account_id=account.id, gmail_message_id="live-1",
            from_email="sender@example.com", subject="Project update",
            body="The project milestone is complete.",
            received_at=datetime.now(timezone.utc),
        )
        db.add(email)
        db.commit()

        status = {"email_ids": [str(email.id)], "complete": True, "error_code": ""}
        monkeypatch.setattr(gmail_mcp, "search_live_mailbox", lambda *a, **k: {
            **status, "source": "gmail_mcp",
        })
        monkeypatch.setattr(server, "search_mailbox", lambda *a, **k: {
            **status, "source": "gmail_mcp",
        })
        from app.database import connection
        monkeypatch.setattr(connection, "SessionLocal", lambda: Session(engine))

        tools = {candidate.name: candidate for candidate in create_gmail_tools(db, user.id)}
        mcp_result = json.loads(tools["search_gmail_mcp"].invoke({"query": "project"}))
        direct_result = json.loads(tools["search_gmail_live"].invoke({"query": "project"}))
        assert mcp_result["source"] == "gmail_mcp"
        assert direct_result["source"] == "gmail_direct"
        assert mcp_result["complete"] and direct_result["complete"]
        assert mcp_result["emails"][0]["email_id"] == str(email.id)
        assert direct_result["emails"][0]["email_id"] == str(email.id)
    engine.dispose()


def test_mcp_transport_failure_uses_direct_gmail_api(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    from app.database import connection
    monkeypatch.setattr(gmail_mcp, "_search", lambda *a, **k: None)
    monkeypatch.setattr(gmail_mcp.asyncio, "run", lambda *a, **k: (_ for _ in ()).throw(
        TimeoutError("MCP transport unavailable")))
    monkeypatch.setattr(connection, "SessionLocal", lambda: Session(engine))
    monkeypatch.setattr(server, "search_mailbox", lambda *args, **kwargs: {
        "email_ids": [], "complete": True, "source": "gmail_mcp", "error_code": "",
    })

    status = gmail_mcp.search_live_mailbox(uuid4(), "latest project mail")
    assert status["source"] == "gmail_direct"
    assert status["complete"] is True
    engine.dispose()


def test_gmail_mcp_server_aliases_exist():
    assert hasattr(server, "search_primary")
    assert server.search_primary is not None

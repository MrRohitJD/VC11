import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

from app.api.routes import rag
from app.rag.activity import emit_activity


def collect(response):
    async def read():
        return "".join([chunk async for chunk in response.body_iterator])
    return asyncio.run(read())


def test_progress_stream_precedes_final_answer_and_uses_its_own_db(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    @contextmanager
    def session():
        yield SimpleNamespace(get=lambda model, uid: user if uid == user.id else None)
    monkeypatch.setattr(rag, "SessionLocal", session)
    def answer(body, scoped_user, db, timezone):
        assert scoped_user.id == user.id
        assert timezone == "Asia/Kolkata"
        emit_activity("keyword_search", "Found 3 candidates", count=3)
        return rag.ChatResponse(answer="Verified answer", intent="RAG_QUERY", citations=[], steps=[], session_id="session", message_id="message")
    monkeypatch.setattr(rag, "chat_with_assistant", answer)
    data = collect(rag.stream_chat(rag.ChatRequest(query="Find interview mail"), user, "Asia/Kolkata"))
    assert data.index("event: activity") < data.index("event: result")
    assert "Found 3 candidates" in data
    assert "Verified answer" in data
    assert "api_key" not in data


def test_stream_errors_do_not_expose_internal_exception_details(monkeypatch):
    @contextmanager
    def session():
        raise RuntimeError("secret credential data")
        yield
    monkeypatch.setattr(rag, "SessionLocal", session)
    data = collect(rag.stream_chat(rag.ChatRequest(query="Find mail"), SimpleNamespace(id=uuid4()), None))
    assert "event: error" in data
    assert "secret credential data" not in data

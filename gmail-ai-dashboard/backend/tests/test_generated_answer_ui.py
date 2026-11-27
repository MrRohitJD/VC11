import json
from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.rag.chains import generate_answer_ui, parse_grounded_answer_output
from app.rag.agent import AgentResponse
from app.rag.retrieval import RetrievedCitation
from app.api.routes import rag as rag_routes


def test_parses_llm_designed_ui_without_replacing_grounded_answer():
    raw = json.dumps({
        "answer": "The bill is due on the 28th. [1]",
        "presentation": {
            "html": '<article><button data-source="1">Electricity bill</button></article>',
            "css": "article { display: grid; }",
            "javascript": "document.querySelector('article').classList.add('ready');",
        },
    })

    answer, presentation = parse_grounded_answer_output(raw, 1)

    assert answer == "The bill is due on the 28th. [1]"
    assert presentation["html"].startswith("<article>")
    assert presentation["css"] == "article { display: grid; }"
    assert presentation["javascript"].startswith("document.querySelector")


def test_rejects_oversized_generated_ui_but_keeps_answer():
    raw = json.dumps({"answer": "A grounded answer.", "presentation": {
        "html": "<div>" + "x" * 16000 + "</div>", "css": "", "javascript": "",
    }})

    assert parse_grounded_answer_output(raw, 1) == ("A grounded answer.", None)


def test_designs_ui_from_final_answer_when_graph_drops_earlier_presentation():
    seen = []

    def reply(prompt):
        seen.append(str(prompt))
        return AIMessage(content=json.dumps({
            "html": '<section><h2>Billing answer</h2><button data-source="1">Open bill</button></section>',
            "css": "section { color: #24315c; }",
            "javascript": "",
        }))

    citation = SimpleNamespace(subject="Electricity bill", sender="billing@example.com",
                               date="2026-11-21", snippet="1,250 INR due on the 28th",
                               key_points=[])
    presentation = generate_answer_ui(
        RunnableLambda(reply), "Search billing emails", "The electricity bill is 1,250 INR.", [citation],
    )

    assert presentation["html"].startswith("<section>")
    assert presentation["css"] == "section { color: #24315c; }"
    assert "The electricity bill is 1,250 INR" in seen[0]
    assert "Electricity bill" in seen[0]


def test_chat_returns_and_persists_ui_for_plain_graph_answer(monkeypatch):
    session_id, message_id = uuid4(), uuid4()
    citation = RetrievedCitation(email_id=str(uuid4()), subject="Electricity bill",
                                 sender="billing@example.com", date="2026-11-21",
                                 snippet="1,250 INR due on the 28th", score=0.9)
    result = AgentResponse(answer="The electricity bill is 1,250 INR.", intent="RAG_QUERY",
                           citations=[citation], session_id=str(session_id), message_id=str(message_id))
    assistant = SimpleNamespace(model=object(), run=lambda **_: result)
    presentation = {"html": "<h2>Billing</h2>", "css": "h2{color:blue}", "javascript": ""}
    message = SimpleNamespace(chat_session_id=session_id, presentation=None)
    commits = []
    db = SimpleNamespace(get=lambda *_: message, commit=lambda: commits.append(True), rollback=lambda: None)
    monkeypatch.setattr(rag_routes, "default_assistant", assistant)
    monkeypatch.setattr(rag_routes, "personal_chat_model", lambda: None)
    monkeypatch.setattr(rag_routes, "generate_answer_ui", lambda *_: presentation)
    monkeypatch.setattr(rag_routes, "log_ai_usage", lambda **_: None)

    response = rag_routes._run_chat(
        rag_routes.ChatRequest(query="Search emails regarding billing"),
        SimpleNamespace(id=uuid4(), is_demo=False), db, None,
    )

    assert response.presentation == presentation
    assert message.presentation == presentation
    assert commits == [True]

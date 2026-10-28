"""Exercise real graph routing and checkpoint persistence without provider calls."""
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from app.rag import graph_agent
from app.rag.retrieval import RetrievedCitation


def test_retrieval_graph_and_persistent_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_agent, "create_gmail_tools", lambda *_: [])
    calls = []

    def answer_query(**kwargs):
        calls.append(kwargs)
        return {"answer": "A verified mailbox answer", "citations": [RetrievedCitation(
            email_id="source-1", subject="Invoice", sender="alice", date="", snippet="evidence", score=1.0,
        )]}

    assistant = SimpleNamespace(
        transform_query=lambda query, history: query,
        route_intent=lambda query: "RAG_QUERY",
        rag_chain=SimpleNamespace(answer_query=answer_query),
        max_agent_iterations=3,
    )
    path = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "account:session"}}
    with SqliteSaver.from_conn_string(path) as saver:
        graph = graph_agent.build_mailbox_graph(
            assistant, db=None, user_id="account", history=[], top_k=7,
            filter_dict={"sender": "alice"}, use_multi_query=True,
            citations=[], checkpointer=saver,
        )
        result = graph.invoke({"query": "Find invoices", "steps": [], "messages": [], "iterations": 0}, config)
        assert result["answer"] == "A verified mailbox answer"
        assert calls[0]["filter_dict"] == {"sender": "alice"}
        assert calls[0]["top_k"] == 7
        assert result["plan"]
        assert result["search_plan"]["strategy"] == "multi_query"
    with SqliteSaver.from_conn_string(path) as reopened:
        checkpoint = reopened.get_tuple(config)
        assert checkpoint.checkpoint["channel_values"]["answer"] == result["answer"]
        assert reopened.get_tuple({"configurable": {"thread_id": "other:session"}}) is None


def test_graph_tool_loop_excludes_fake_send(monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain_core.tools import tool

    @tool
    def send_email(approval_id: str) -> str:
        """An unavailable fake delivery tool."""
        raise AssertionError("Must never execute")

    monkeypatch.setattr(graph_agent, "create_gmail_tools", lambda *_: [send_email])
    class Model:
        def bind_tools(self, tools):
            assert tools == []
            return self

        def invoke(self, messages):
            return AIMessage(content="Sending is unavailable.")

    assistant = SimpleNamespace(model=Model(), max_agent_iterations=3,
        transform_query=lambda query, history: query,
        route_intent=lambda query: "ACTION_SEND")
    graph = graph_agent.build_mailbox_graph(
        assistant, db=None, user_id="account", history=[], top_k=5,
        filter_dict=None, use_multi_query=False, citations=[], checkpointer=InMemorySaver())
    result = graph.invoke({"query": "send email", "messages": [], "steps": [], "iterations": 0},
                         {"configurable": {"thread_id": "account:session"}})
    assert result["answer"] == "Sending is unavailable."

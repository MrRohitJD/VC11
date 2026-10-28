"""Focused retrieval contracts that require no network or production mailbox."""
import json
from types import SimpleNamespace
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models.email import Base, Email, EmailLabel, GmailAccount, Label, User
from app.models.rag import EmailChunk, EmailEmbedding
from app.rag.agent import extract_citations_from_tool_observation
from app.rag import graph_agent
from app.rag.retrieval import PostgresSemanticRetriever, RetrievedCitation, default_retriever
from app.rag.tools import create_gmail_tools
from app.rag.pgvector_store import PostgresVectorStoreManager
from app.rag.chains import GroundedRAGChain
from app.rag.relevance import RelevanceResult
from scripts.evaluate_retrieval import ranking_metrics


def _graph_with_model(monkeypatch, responses, search_tool, *, effective_query=None,
                      history=None, chain_answer=None):
    calls = []
    search_tools = search_tool if isinstance(search_tool, list) else [search_tool]

    class Model:
        def bind_tools(self, tools):
            assert all(candidate in tools for candidate in search_tools)
            return self

        def invoke(self, messages):
            calls.append(messages)
            return responses.pop(0)

    monkeypatch.setattr(graph_agent, "create_gmail_tools", lambda *_: search_tools)
    assistant = SimpleNamespace(
        model=Model(), max_agent_iterations=4,
        transform_query=lambda query, history: query,
        route_intent=lambda query: "RAG_QUERY",
        rag_chain=SimpleNamespace(answer_query=chain_answer or (lambda **_: {
            "answer": "I could not verify any matching emails after the broader search.",
            "citations": [], "search_plan": {"strategy": "grounded_chain"},
        })),
    )
    if effective_query:
        assistant.interpret_request = lambda query, messages: SimpleNamespace(
            intent="RAG_QUERY", effective_query=effective_query,
            draft=None, clarification_question=None,
        )
    graph = graph_agent.build_mailbox_graph(
        assistant, db=None, user_id="account", history=history or [], top_k=5,
        filter_dict=None, use_multi_query=None, citations=[], checkpointer=InMemorySaver(),
    )
    return graph, calls


def test_model_selects_read_tool_and_uses_its_evidence(monkeypatch):
    searched = []

    @tool
    def search_emails_exact(phrase: str) -> str:
        """Find a literal phrase in authorized email records."""
        searched.append(phrase)
        return json.dumps([{"email_id": "message-1", "subject": "Project", "sender": "alice",
                            "snippet": "The release code is ZX-41."}])

    graph, calls = _graph_with_model(monkeypatch, [
        AIMessage(content="", tool_calls=[{"name": "search_emails_exact",
                                            "args": {"phrase": "ZX-41"}, "id": "call-1"}]),
        AIMessage(content="The release code is ZX-41."),
    ], search_emails_exact)
    result = graph.invoke({"query": "What was the release code?", "messages": [],
                           "steps": [], "iterations": 0},
                          {"configurable": {"thread_id": "user:tool-selection"}})
    assert result["answer"] == "The release code is ZX-41."
    assert searched == ["ZX-41"]
    assert result["search_plan"]["providers"] == ["search_emails_exact"]
    assert len(calls) == 2


def test_repeated_identical_read_call_uses_request_cache(monkeypatch):
    invocations = []

    @tool
    def search_emails_exact(phrase: str) -> str:
        """Find a literal phrase in authorized email records."""
        invocations.append(phrase)
        return json.dumps([{"email_id": "message-1", "subject": "Project", "snippet": phrase}])

    call = lambda id: AIMessage(content="", tool_calls=[
        {"name": "search_emails_exact", "args": {"phrase": "ZX-41"}, "id": id}])
    graph, _ = _graph_with_model(monkeypatch, [call("call-1"), call("call-2"),
                                               AIMessage(content="Found ZX-41.")], search_emails_exact)
    result = graph.invoke({"query": "Find ZX-41", "messages": [], "steps": [], "iterations": 0},
                          {"configurable": {"thread_id": "user:cached-search"}})
    assert result["answer"] == "Found ZX-41."
    assert invocations == ["ZX-41"]


def test_read_answer_without_email_evidence_is_rejected(monkeypatch):
    @tool
    def search_emails_exact(phrase: str) -> str:
        """Find a literal phrase in authorized email records."""
        raise AssertionError("Should not be called")

    graph, _ = _graph_with_model(monkeypatch, [
        AIMessage(content="I saw an email saying the deadline is Friday."),
    ], search_emails_exact)
    result = graph.invoke({"query": "When is the deadline?", "messages": [],
                           "steps": [], "iterations": 0},
                          {"configurable": {"thread_id": "user:no-evidence"}})
    assert result["answer"] == "I could not verify any matching emails after the broader search."


def test_empty_search_cannot_be_reported_as_a_found_email(monkeypatch):
    @tool
    def search_emails_exact(phrase: str) -> str:
        """Find a literal phrase in authorized email records."""
        return "[]"

    graph, _ = _graph_with_model(monkeypatch, [
        AIMessage(content="", tool_calls=[{"name": "search_emails_exact",
                                            "args": {"phrase": "missing"}, "id": "call-1"}]),
        AIMessage(content="I found the missing email."),
    ], search_emails_exact)
    result = graph.invoke({"query": "Find missing", "messages": [], "steps": [], "iterations": 0},
                          {"configurable": {"thread_id": "user:empty-search"}})
    assert result["answer"] == "I could not verify any matching emails after the broader search."


def test_empty_narrow_search_recovers_full_parcel_request(monkeypatch):
    @tool
    def search_emails_exact(phrase: str) -> str:
        """Find a literal phrase in authorized email records."""
        return "[]"

    question = ("Find all my parcel delivery emails and summarize which packages "
                "have been shipped, delivered, delayed, or are still on the way.")
    recovered_queries = []

    def grounded_answer(**kwargs):
        recovered_queries.append(kwargs["query"])
        return {"answer": "The sample package is out for delivery.",
                "citations": [RetrievedCitation(
                    email_id="parcel-1", subject="Package is out for delivery",
                    sender="tracking@parcel.example", date="", snippet="Out for delivery today.",
                    score=1.0,
                )], "search_plan": {"strategy": "exhaustive_mailbox"}}

    graph, _ = _graph_with_model(monkeypatch, [
        AIMessage(content="", tool_calls=[{"name": "search_emails_exact",
                                            "args": {"phrase": "parcel delivery"}, "id": "call-1"}]),
        AIMessage(content="No parcel messages exist."),
    ], search_emails_exact, chain_answer=grounded_answer)
    result = graph.invoke({"query": question, "messages": [], "steps": [], "iterations": 0},
                          {"configurable": {"thread_id": "user:parcel-recovery"}})
    assert recovered_queries == [question]
    assert result["answer"] == "The sample package is out for delivery."
    assert result["search_plan"]["strategy"] == "exhaustive_mailbox"


def test_model_can_combine_semantic_and_exact_results(monkeypatch):
    @tool
    def search_emails(query: str) -> str:
        """Find meaning-based matches."""
        return json.dumps([{"email_id": "message-1", "subject": "Launch plan",
                            "snippet": "The rollout begins Monday."}])

    @tool
    def search_emails_exact(phrase: str) -> str:
        """Find literal matches."""
        return json.dumps([{"email_id": "message-2", "subject": "Tracking",
                            "snippet": "Reference ZX-41 closes Friday."}])

    graph, calls = _graph_with_model(monkeypatch, [
        AIMessage(content="", tool_calls=[
            {"name": "search_emails", "args": {"query": "rollout"}, "id": "call-1"},
            {"name": "search_emails_exact", "args": {"phrase": "ZX-41"}, "id": "call-2"},
        ]),
        AIMessage(content="The rollout starts Monday and ZX-41 closes Friday."),
    ], [search_emails, search_emails_exact])
    result = graph.invoke({"query": "Compare the rollout and ZX-41", "messages": [],
                           "steps": [], "iterations": 0},
                          {"configurable": {"thread_id": "user:multi-tool"}})
    assert result["search_plan"]["providers"] == ["search_emails", "search_emails_exact"]
    assert len(calls) == 2
    assert result["answer"].startswith("The rollout")


def test_followup_uses_interpreted_standalone_query(monkeypatch):
    @tool
    def search_emails(query: str) -> str:
        """Find meaning-based matches."""
        return json.dumps([{"email_id": "message-1", "subject": "Release",
                            "snippet": "The release is Friday."}])

    standalone = "When is the product release?"
    graph, calls = _graph_with_model(monkeypatch, [
        AIMessage(content="", tool_calls=[{"name": "search_emails",
                                            "args": {"query": standalone}, "id": "call-1"}]),
        AIMessage(content="The release is Friday."),
    ], search_emails, effective_query=standalone,
       history=[HumanMessage(content="Tell me about the product release")])
    result = graph.invoke({"query": "When is it?", "messages": [], "steps": [], "iterations": 0},
                          {"configurable": {"thread_id": "user:followup"}})
    assert result["answer"] == "The release is Friday."
    assert calls[0][-1].content == standalone


def test_grounded_answer_tool_finishes_without_second_synthesis_call(monkeypatch):
    monkeypatch.setattr(graph_agent, "create_gmail_tools", lambda *_: [])
    model_calls = []
    chain_calls = []

    class Model:
        def bind_tools(self, tools):
            assert [candidate.name for candidate in tools] == ["answer_mailbox_question"]
            return self

        def invoke(self, messages):
            model_calls.append(messages)
            return AIMessage(content="", tool_calls=[{
                "name": "answer_mailbox_question", "args": {"query": "Summarize the release"}, "id": "call-1",
            }])

    def answer_query(**kwargs):
        chain_calls.append(kwargs)
        return {"answer": "The release is Friday.", "citations": [RetrievedCitation(
            email_id="message-1", subject="Release", sender="alice",
            date="", snippet="The release is Friday.", score=.9,
        )], "search_plan": {"strategy": "vector"}}

    assistant = SimpleNamespace(model=Model(), rag_chain=SimpleNamespace(answer_query=answer_query),
                                max_agent_iterations=3, transform_query=lambda query, history: query,
                                route_intent=lambda query: "SUMMARIZE")
    user_id = uuid4()
    db = SimpleNamespace(get=lambda *_: None)
    graph = graph_agent.build_mailbox_graph(
        assistant, db=db, user_id=user_id, history=[], top_k=5,
        filter_dict=None, use_multi_query=None, citations=[], checkpointer=InMemorySaver(),
    )
    result = graph.invoke({"query": "Summarize the release", "messages": [], "steps": [],
                           "iterations": 0}, {"configurable": {"thread_id": "user:grounded"}})
    assert result["answer"] == "The release is Friday."
    assert result["search_plan"] == {"strategy": "vector"}
    assert len(model_calls) == len(chain_calls) == 1


def test_semantic_candidates_are_distinct_emails_before_top_k():
    user_id = str(uuid4())
    first, second = str(uuid4()), str(uuid4())

    class Vectors:
        def __init__(self):
            self.requested = []

        def build_filter(self, **kwargs):
            return kwargs

        def search_with_scores(self, **kwargs):
            self.requested.append(kwargs["top_k"])
            return [(Document(page_content=f"chunk {index}", metadata={"email_id": email_id}), score)
                    for index, (email_id, score) in enumerate([
                        (first, .96), (first, .95), (first, .94), (second, .90)])]

    vectors = Vectors()
    retriever = PostgresSemanticRetriever(user_id=user_id, vector_manager=vectors, top_k=2)
    docs = retriever.invoke("different wording for project deadline")
    assert [d.metadata["email_id"] for d in docs] == [first, second]
    assert vectors.requested == [2 * retriever.candidate_multiplier]
    baseline = ranking_metrics([first], {first, second}, 2)
    improved = ranking_metrics([doc.metadata["email_id"] for doc in docs], {first, second}, 2)
    assert baseline == (.5, .5, 1.0)
    assert improved == (1.0, 1.0, 1.0)


def test_exact_search_stays_within_authenticated_user():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        first_user, second_user = uuid4(), uuid4()
        first_account, second_account = uuid4(), uuid4()
        db.add_all([
            User(id=first_user, google_id="first", email="first@example.com"),
            User(id=second_user, google_id="second", email="second@example.com"),
            GmailAccount(id=first_account, user_id=first_user, email="first@example.com", access_token="x"),
            GmailAccount(id=second_account, user_id=second_user, email="second@example.com", access_token="x"),
        ])
        email_ids = [uuid4(), uuid4()]
        db.add_all([
            Email(id=email_ids[0], gmail_account_id=first_account, gmail_message_id="one",
                  subject="Private launch", snippet="An unrelated preview", body="The code is ZX-41.",
                  from_email="a@example.com"),
            Email(id=email_ids[1], gmail_account_id=second_account, gmail_message_id="two",
                  subject="Private ZX-41", body="The code is ZX-41.", from_email="b@example.com"),
        ])
        db.commit()
        first_tool = next(t for t in create_gmail_tools(db, first_user) if t.name == "search_emails_exact")
        second_tool = next(t for t in create_gmail_tools(db, second_user) if t.name == "search_emails_exact")
        first_results = json.loads(first_tool.invoke({"phrase": "ZX-41"}))
        second_results = json.loads(second_tool.invoke({"phrase": "ZX-41"}))
        assert [r["email_id"] for r in first_results] == [str(email_ids[0])]
        assert [r["email_id"] for r in second_results] == [str(email_ids[1])]
        assert "ZX-41" in first_results[0]["snippet"]
        assert default_retriever.retrieve_structured_sql(db, first_user, email_id="invalid") == []
    engine.dispose()


def test_tool_citation_preserves_thread_and_unknown_score():
    citation = extract_citations_from_tool_observation(json.dumps([{
        "email_id": "message-1", "subject": "Update", "thread_id": "thread-1", "score": 0.0,
    }]))[0]
    assert citation.thread_id == "thread-1"
    assert citation.score == 0.0


def test_vector_scope_stays_in_sql_for_plain_queries(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    class Embeddings:
        def embed_query(self, query):
            return [1.0] * 3072

    with Session(engine) as db:
        user_id, other_user = uuid4(), uuid4()
        account, other_account = uuid4(), uuid4()
        db.add_all([
            User(id=user_id, google_id="vector-owner", email="vector-owner@example.com"),
            User(id=other_user, google_id="vector-other", email="vector-other@example.com"),
            GmailAccount(id=account, user_id=user_id, email="vector-owner@example.com", access_token="x"),
            GmailAccount(id=other_account, user_id=other_user, email="vector-other@example.com", access_token="x"),
        ])
        allowed, spam, foreign = [uuid4() for _ in range(3)]
        for email_id, gmail_account_id in [(allowed, account), (spam, account), (foreign, other_account)]:
            db.add(Email(id=email_id, gmail_account_id=gmail_account_id,
                         gmail_message_id=str(email_id), subject="Project plan", body="Project details"))
            chunk_id = uuid4()
            db.add(EmailChunk(id=chunk_id, email_id=email_id, chunk_index=0,
                              chunk_text="Project details", content_hash=str(email_id)))
            db.add(EmailEmbedding(id=uuid4(), chunk_id=chunk_id, email_id=email_id,
                                  embedding=[1.0] * 3072, embedding_dimensions=3072,
                                  chroma_record_id=str(chunk_id), content_hash=str(email_id),
                                  status="INDEXED"))
        label_id = uuid4()
        db.add(Label(id=label_id, gmail_account_id=account, label_id="SPAM", name="SPAM"))
        db.add(EmailLabel(email_id=spam, gmail_account_id=account, label_id=label_id))
        db.commit()

        from app.rag import retrieval
        monkeypatch.setattr(retrieval, "mailbox_email_ids", lambda *_: (_ for _ in ()).throw(
            AssertionError("Plain vector search should use SQL scope")))
        manager = PostgresVectorStoreManager(embedding_function=Embeddings(), db_session=db)
        docs = PostgresSemanticRetriever(user_id=str(user_id), db_session=db,
                                         vector_manager=manager, top_k=5).invoke("Project plan")
        assert [doc.metadata["email_id"] for doc in docs] == [str(allowed)]
    engine.dispose()


def test_exhaustive_grounded_search_finds_unindexed_parcel_email(monkeypatch):
    from app.rag import chains

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user_id, other_user = uuid4(), uuid4()
        account_id, other_account = uuid4(), uuid4()
        db.add_all([
            User(id=user_id, google_id="parcel-owner", email="parcel-owner@example.com", is_demo=True),
            User(id=other_user, google_id="parcel-other", email="parcel-other@example.com", is_demo=True),
            GmailAccount(id=account_id, user_id=user_id, email="parcel-owner@example.com", access_token="x"),
            GmailAccount(id=other_account, user_id=other_user, email="parcel-other@example.com", access_token="x"),
        ])
        parcel_id = uuid4()
        db.add_all([
            Email(id=parcel_id, gmail_account_id=account_id, gmail_message_id="parcel-owner-1",
                  subject="Package is out for delivery", body="Your order is out for delivery today."),
            Email(id=uuid4(), gmail_account_id=other_account, gmail_message_id="parcel-other-1",
                  subject="Another package", body="Your other order was delivered."),
        ])
        updates_label = uuid4()
        db.add(Label(id=updates_label, gmail_account_id=account_id,
                     label_id="CATEGORY_UPDATES", name="Updates"))
        db.add(EmailLabel(email_id=parcel_id, gmail_account_id=account_id, label_id=updates_label))
        db.commit()

        def validate(documents, query, model, **kwargs):
            matching = [doc for doc in documents if doc.metadata["email_id"] == str(parcel_id)]
            return RelevanceResult(documents=matching, evaluated=len(documents))

        monkeypatch.setattr(chains, "validate_email_relevance", validate)
        monkeypatch.setattr(chains, "generate_grounded_answer", lambda *args: "The package is out for delivery.")
        question = ("Find all my parcel delivery emails and summarize which packages "
                    "have been shipped, delivered, delayed, or are still on the way.")
        result = GroundedRAGChain(model=FakeListChatModel(responses=["unused"])).answer_query(
            db=db, query=question, user_id=user_id,
        )
        assert result["retrieval_strategy"] == "exhaustive_mailbox"
        assert [citation.email_id for citation in result["citations"]] == [str(parcel_id)]
        assert "out for delivery" in result["answer"]
    engine.dispose()

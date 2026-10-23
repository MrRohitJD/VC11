"""Regression coverage for Primary-only assistant answers and tool access."""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import app.models.llm  # Register all related ORM models.
from app.models.email import Base, Email, EmailLabel, GmailAccount, Label, User
from app.models.rag import Attachment
from app.rag.agent import GmailAIAssistant
from app.rag.chains import GroundedRAGChain
from app.rag.embeddings import DeterministicOfflineEmbeddings
from app.rag.mailbox_scope import NON_PRIMARY_LABELS, primary_email_ids
from app.rag.retrieval import ChromaSemanticRetriever, GmailCustomHybridRetriever
from app.rag.tools import create_gmail_tools
from app.rag.vectorstore import ChromaVectorStoreManager


def test_all_date_summary_uses_every_primary_record_without_embedding_or_relevance(mailbox, monkeypatch):
    from app.core.config import settings
    import app.rag.chains as chains
    from app.rag.query_constraints import mailbox_time_context
    db, user_id, *rest = mailbox
    monkeypatch.setattr(settings, "gmail_mcp_enabled", False)
    monkeypatch.setattr(chains, "validate_email_relevance", lambda *a, **k: pytest.fail("Date-only requests need no semantic validator"))
    monkeypatch.setattr(ChromaSemanticRetriever, "_get_relevant_documents", lambda *a, **k: pytest.fail("Exhaustive search must not depend on embeddings"))
    chain = GroundedRAGChain(model=FakeListChatModel(responses=["Summary."]))
    with mailbox_time_context("Asia/Kolkata"):
        result = chain.answer_query(db, "Summarize all incoming emails this month", user_id, top_k=1)
    assert result["context_count"] == 2
    assert result["retrieval_strategy"] == "exhaustive_primary"


def planning_model():
    """Validate candidate identities dynamically, then generate a short answer."""
    def respond(value):
        messages = value.to_messages() if hasattr(value, "to_messages") else value
        try:
            data = json.loads(messages[-1].content)
        except (ValueError, TypeError):
            data = {}
        if "candidates" in data:
            return AIMessage(content=json.dumps({"decisions": [{
                "email_id": c["email_id"], "relevant": True,
                "reason": "Contains planning details.", "key_points": ["Planning details."],
            } for c in data["candidates"]]}))
        return AIMessage(content="Planning details.")
    return RunnableLambda(respond)


@pytest.fixture
def mailbox(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = Session(engine)
    users = [User(google_id=f"g{i}", email=f"user{i}@example.com") for i in range(2)]
    db.add_all(users)
    db.flush()
    accounts = [GmailAccount(user_id=u.id, email=u.email, access_token="test") for u in users]
    db.add_all(accounts)
    db.flush()
    labels = {}
    emails = {}

    def add_email(name, label_ids, account=accounts[0]):
        email = Email(
            gmail_account_id=account.id, gmail_message_id=name, subject=f"Planning {name}",
            body=f"Planning details for {name}.", snippet=f"Planning details for {name}.",
            from_email="sender@example.com", thread_id="shared-thread",
            received_at=datetime(2026, 10, 18, tzinfo=timezone.utc),
        )
        db.add(email)
        db.flush()
        for label_id in label_ids:
            key = (account.id, label_id)
            if key not in labels:
                labels[key] = Label(gmail_account_id=account.id, label_id=label_id, name=label_id)
                db.add(labels[key])
                db.flush()
            db.add(EmailLabel(email_id=email.id, label_id=labels[key].id, gmail_account_id=account.id))
        db.add(Attachment(email_id=email.id, filename=f"{name}.txt", extracted_text=f"Content for {name}"))
        emails[name] = email

    # Index other categories first and with an exact query match to exercise
    # pre-filtering before top_k, rather than dropping them after ranking.
    for category in NON_PRIMARY_LABELS:
        add_email(category, ["INBOX", category])
    add_email("personal", ["CATEGORY_PERSONAL"])
    add_email("inbox", ["INBOX", "IMPORTANT"])
    add_email("unknown", [])
    add_email("custom", ["Label_primary"])
    labels[(accounts[0].id, "Label_primary")].name = "Primary"
    add_email("sent", ["SENT"])
    for excluded in ("SPAM", "TRASH", "DRAFT", "CATEGORY_PROMOTIONS"):
        add_email(f"conflict-{excluded}", ["CATEGORY_PERSONAL", excluded])
    add_email("other-user", ["CATEGORY_PERSONAL"], accounts[1])
    db.commit()

    vstore = ChromaVectorStoreManager(
        persist_directory=str(tmp_path), collection_name=f"primary_{uuid4().hex}",
        embedding_model=DeterministicOfflineEmbeddings(),
    )
    vstore.add_documents([
        Document(page_content=e.body, metadata={
            "user_id": str(accounts[1].user_id if name == "other-user" else users[0].id),
            "email_id": str(e.id), "subject": e.subject, "thread_id": e.thread_id,
            # Deliberately no category metadata: existing vectors must work.
        }) for name, e in emails.items()
    ])
    monkeypatch.setattr("app.rag.retrieval.default_chroma_manager", vstore)
    yield db, users[0].id, emails, vstore
    db.close()
    engine.dispose()


def test_scope_uses_gmail_labels_and_user_isolation(mailbox):
    db, user_id, emails, _ = mailbox
    assert set(primary_email_ids(db, user_id)) == {str(emails[n].id) for n in ("personal", "inbox")}


@pytest.mark.parametrize("search_type", ["similarity", "mmr"])
def test_vector_search_filters_before_ranking_and_honors_relabeling(mailbox, search_type):
    db, user_id, emails, vstore = mailbox
    retriever = ChromaSemanticRetriever(
        user_id=str(user_id), db_session=db, primary_only=True, chroma_manager=vstore,
        top_k=1, search_type=search_type,
    )
    docs = retriever.invoke("Planning details for CATEGORY_PROMOTIONS.")
    assert len(docs) == 1
    assert docs[0].metadata["email_id"] in primary_email_ids(db, user_id)

    # Changing synced labels must take effect without regenerating embeddings.
    for name in ("personal", "inbox"):
        for link in list(emails[name].labels):
            db.delete(link)
    db.commit()
    assert retriever.invoke("Planning") == []


@pytest.mark.parametrize("vector_available", [True, False])
def test_hybrid_and_sql_fallback_remain_primary_only(mailbox, monkeypatch, vector_available):
    db, user_id, emails, vstore = mailbox
    if not vector_available:
        def unavailable(**kwargs):
            raise RuntimeError("Vector search unavailable")
        monkeypatch.setattr(vstore, "search_with_scores", unavailable)
    retriever = GmailCustomHybridRetriever(
        user_id=str(user_id), db_session=db, primary_only=True, chroma_manager=vstore, top_k=20,
    )
    docs = retriever.invoke("Planning")
    assert {d.metadata["email_id"] for d in docs} == {str(emails[n].id) for n in ("personal", "inbox")}


@pytest.mark.parametrize("multi_query", [False, True])
def test_answer_chain_only_cites_primary_even_with_override_attempt(mailbox, multi_query):
    db, user_id, emails, _ = mailbox
    chain = GroundedRAGChain(model=planning_model())
    result = chain.answer_query(
        db=db, query="Planning", user_id=user_id, use_multi_query=multi_query,
        filter_dict={"primary_only": False},
    )
    assert result["citations"]
    assert {c.email_id for c in result["citations"]} <= {str(emails[n].id) for n in ("personal", "inbox")}


def test_no_match_does_not_broaden_search(mailbox):
    db, user_id, emails, _ = mailbox
    model = FakeListChatModel(responses=["No matching emails found in Primary."])
    chain = GroundedRAGChain(model=model)
    result = chain.answer_query(
        db=db, query="Planning", user_id=user_id,
        filter_dict={"gmail_message_id": emails["CATEGORY_PROMOTIONS"].gmail_message_id},
    )
    assert result["citations"] == []
    assert "Primary" in result["answer"]


def test_agent_sql_fallback_remains_primary_only(mailbox, monkeypatch):
    db, user_id, emails, _ = mailbox
    agent = GmailAIAssistant(model=FakeListChatModel(responses=["Planning details."]))
    monkeypatch.setattr(agent, "execute_react_agent", lambda *args: ("", [], []))
    _, citations, _ = agent.execute_with_fallback("Planning", db, user_id, [])
    assert {c.email_id for c in citations} == {str(emails[n].id) for n in ("personal", "inbox")}


def test_tools_cannot_read_non_primary_content(mailbox):
    db, user_id, emails, _ = mailbox
    tool_map = {t.name: t for t in create_gmail_tools(db, user_id)}
    for name in ("CATEGORY_PROMOTIONS", "other-user"):
        eid = str(emails[name].id)
        assert "not found" in tool_map["get_email_by_id"].invoke({"email_id": eid})
        assert "not found" in tool_map["get_email_summary"].invoke({"email_id": eid})
        assert "No attachments" in tool_map["get_email_attachments"].invoke({"email_id": eid})

    primary_id = str(emails["personal"].id)
    assert "personal.txt" in tool_map["get_email_attachments"].invoke({"email_id": primary_id})
    assert "Planning personal" in tool_map["get_email_by_id"].invoke({"email_id": primary_id})
    for tool_name, args in [
        ("search_emails", {"query": "Planning"}),
        ("search_email_threads", {"thread_id": "shared-thread"}),
        ("filter_emails_hybrid", {"query": "Planning"}),
        ("filter_emails_by_sender", {"sender": "sender@example.com"}),
        ("filter_emails_by_date", {"start_date": "2026-10-17", "end_date": "2026-10-19"}),
        ("filter_emails_by_label", {"label": "Primary"}),
        ("retrieve_email_context", {"query": "Planning"}),
        ("search_emails_multi_query", {"query": "Planning"}),
    ]:
        rows = json.loads(tool_map[tool_name].invoke(args))
        assert rows
        assert {r["email_id"] for r in rows} <= {str(emails[n].id) for n in ("personal", "inbox")}
    assert json.loads(tool_map["filter_emails_by_label"].invoke({"label": "CATEGORY_PROMOTIONS"})) == []


def test_primary_scope_requires_authoritative_labels():
    retriever = ChromaSemanticRetriever(user_id=str(uuid4()), primary_only=True)
    with pytest.raises(ValueError, match="database session"):
        retriever.invoke("Planning")


def test_billing_question_cannot_retrieve_pending_interview_documents(mailbox):
    db, user_id, emails, _ = mailbox
    email = emails["personal"]
    email.subject = "Urgent pending interview documents"
    email.body = "Some documents are still pending. Your interview is scheduled tomorrow."
    db.commit()
    query = "What are my pending bills and invoices due this month?"
    chain = GroundedRAGChain(model=FakeListChatModel(responses=["Unrelated response must not be generated."]))
    result = chain.answer_query(db=db, query=query, user_id=user_id)
    assert result["citations"] == []
    assert "could not verify" in result["answer"]
    assert "interview" not in result["answer"]
    assert result["rewritten_query"] == query


@pytest.mark.parametrize("vector_available", [True, False])
def test_bills_use_due_month_not_received_month(mailbox, monkeypatch, vector_available):
    from app.rag.query_constraints import MAILBOX_TIMEZONE
    from app.rag.retrieval import default_retriever
    db, user_id, emails, vstore = mailbox
    now = datetime.now(MAILBOX_TIMEZONE)
    due_date = f"{now.year}-{now.month:02d}-25"
    bill = emails["personal"]
    bill.subject = "Electricity invoice"
    bill.body = f"Your unpaid electricity bill has an amount due of INR 1,200. Payment due on {due_date}."
    bill.from_email = "Utility Billing <billing@example.com>"
    bill.received_at = datetime(now.year - 1, 12, 1)
    wrong = emails["inbox"]
    wrong.subject = "Historic unpaid invoice"
    wrong.body = f"Invoice amount due INR 500. Payment due on {now.year - 1}-{now.month:02d}-25."
    wrong.received_at = now
    db.commit()
    if not vector_available:
        def unavailable(**kwargs):
            raise RuntimeError("Vector service unavailable")
        monkeypatch.setattr(vstore, "search_with_scores", unavailable)
    query = "What are my pending bills and invoices due this month?"
    retriever = GmailCustomHybridRetriever(
        user_id=str(user_id), db_session=db, primary_only=True, chroma_manager=vstore,
    )
    docs = retriever.invoke(query)
    assert {doc.metadata["email_id"] for doc in docs} == {str(bill.id)}
    assert docs[0].metadata["sender"] == bill.from_email
    assert docs[0].metadata["received_at"].startswith(str(now.year - 1))
    citations = default_retriever.retrieve_structured_sql(db, user_id, keyword=query, primary_only=True)
    assert [c.email_id for c in citations] == [str(bill.id)]


def test_source_endpoint_returns_full_envelope_and_html(mailbox):
    from fastapi import HTTPException
    from app.api.routes.rag import get_source_email_details
    db, user_id, emails, _ = mailbox
    email = emails["personal"]
    email.from_email = "Billing Team <billing@example.com>"
    email.to_email = "owner@example.com"
    email.cc = "accounts@example.com"
    email.body = "Full message line one.\n\nFull message beyond citation preview."
    email.body_html = "<p>Full <strong>formatted</strong> message.</p>"
    user = db.get(User, user_id)
    result = get_source_email_details(str(email.id), user, db)
    assert result["from_email"] == "billing@example.com"
    assert result["sender_name"] == "Billing Team"
    assert result["to"] == email.to_email
    assert result["cc"] == email.cc
    assert result["body"] == email.body
    assert result["body_html"] == email.body_html
    assert result["received_at"].endswith("+00:00")
    assert result["attachments"][0]["filename"] == "personal.txt"
    with pytest.raises(HTTPException) as exc:
        get_source_email_details(str(emails["other-user"].id), user, db)
    assert exc.value.status_code == 404


def test_html_only_source_has_readable_text_fallback(mailbox):
    from app.api.routes.rag import get_source_email_details
    db, user_id, emails, _ = mailbox
    email = emails["personal"]
    email.body = None
    email.body_html = "<p>HTML-only message body.</p>"
    result = get_source_email_details(str(email.id), db.get(User, user_id), db)
    assert "HTML-only message body." in result["body"]


def test_agent_new_bills_question_ignores_prior_interview_session(mailbox):
    from app.rag.memory import get_or_create_chat_session
    from app.models.rag import ChatMessage
    db, user_id, _, _ = mailbox
    session = get_or_create_chat_session(db, user_id)
    db.add(ChatMessage(chat_session_id=session.id, role="user", content="Find interview related emails"))
    db.commit()
    query = "What are my pending bills and invoices due this month?"
    agent = GmailAIAssistant(model=FakeListChatModel(responses=["RAG_QUERY"]))
    result = agent.run(query, db, user_id, session_id=session.id)
    assert result.citations == []
    assert "interview" not in result.answer.lower()
    assert result.steps[-1]["arguments"]["query"] == query


def interview_model(observed):
    def respond(value):
        messages = value.to_messages() if hasattr(value, "to_messages") else value
        try:
            payload = json.loads(messages[-1].content)
        except ValueError:
            return AIMessage(content="Only the matching interview invitations are listed.")
        observed.extend(payload.get("candidates", []))
        decisions = []
        for c in payload.get("candidates", []):
            relevant = "invited to your technical interview" in c["body"]
            decisions.append({"email_id": c["email_id"], "relevant": relevant,
                              "reason": "Direct interview invitation." if relevant else "Generic HR responsibilities.",
                              "key_points": ["You are invited to your technical interview."] if relevant else []})
        return AIMessage(content=json.dumps({"decisions": decisions}))
    return RunnableLambda(respond)


@pytest.mark.parametrize("multi_query", [False, True])
def test_interview_month_scope_and_llm_selection_reject_old_and_generic_mail(mailbox, multi_query):
    from app.rag.query_constraints import mailbox_now
    db, user_id, emails, vstore = mailbox
    now = mailbox_now().astimezone(timezone.utc)
    invitation = emails["personal"]
    invitation.subject = "Technical interview confirmation"
    invitation.body = "You are invited to your technical interview. Please confirm your slot."
    invitation.from_email = "Recruiter <recruiter@example.com>"
    invitation.received_at = now
    advertisement = emails["inbox"]
    advertisement.subject = "HR internship advertisement"
    advertisement.body = "Responsibilities include coordinating interview schedules and recruitment."
    advertisement.received_at = now
    old = Email(gmail_account_id=invitation.gmail_account_id, gmail_message_id="old-interview",
                subject="Interview slot confirmation", body="You are invited to your technical interview in 2024.",
                received_at=datetime(2024, 7, 26))
    db.add(old)
    db.flush()
    db.add(EmailLabel(email_id=old.id, label_id=invitation.labels[0].label_id,
                      gmail_account_id=old.gmail_account_id))
    db.commit()
    vstore.add_documents([Document(page_content=old.body, metadata={
        "user_id": str(user_id), "email_id": str(old.id), "subject": old.subject,
    })])
    observed = []
    chain = GroundedRAGChain(model=interview_model(observed))
    result = chain.answer_query(db, "Find all interview related emails of this month", user_id,
                                use_multi_query=multi_query)
    assert result["relevance_validated"]
    assert [c.email_id for c in result["citations"]] == [str(invitation.id)]
    assert all(c["email_id"] != str(old.id) for c in observed)
    candidate = next(c for c in observed if c["email_id"] == str(invitation.id))
    assert candidate["sender_email"] == "recruiter@example.com"
    assert candidate["sender_name"] == "Recruiter"
    assert candidate["received_at"].endswith("+00:00")
    assert candidate["body"] == invitation.body


def test_find_all_keeps_more_than_five_validated_sources(mailbox):
    from app.rag.query_constraints import mailbox_now
    db, user_id, emails, _ = mailbox
    invitation = emails["personal"]
    invitation.body = "You are invited to your technical interview."
    invitation.received_at = mailbox_now().astimezone(timezone.utc)
    for index in range(6):
        email = Email(gmail_account_id=invitation.gmail_account_id, gmail_message_id=f"current-interview-{index}",
                      subject=f"Technical interview invitation {index}", body=invitation.body,
                      received_at=invitation.received_at)
        db.add(email)
        db.flush()
        db.add(EmailLabel(email_id=email.id, label_id=invitation.labels[0].label_id,
                          gmail_account_id=email.gmail_account_id))
    db.commit()
    agent = GmailAIAssistant(model=interview_model([]))
    result = agent.run("Find all interview related emails of this month", db, user_id, top_k=5)
    assert len(result.citations) == 7


def test_llm_failure_does_not_display_keyword_matches_as_validated_results(mailbox):
    db, user_id, _, _ = mailbox
    chain = GroundedRAGChain(model=FakeListChatModel(responses=["An unstructured keyword summary."]))
    result = chain.answer_query(db, "Planning", user_id)
    assert result["relevance_validated"] is False
    assert result["citations"] == []
    assert "could not validate" in result["answer"]


def test_sender_listing_works_without_any_ai_and_ignores_body_mentions(mailbox):
    db, user_id, emails, _ = mailbox
    emails['personal'].from_email = 'ChatGPT <noreply@example.com>'
    emails['inbox'].body = 'An article about ChatGPT.'
    db.commit()
    def unavailable(_):
        pytest.fail('Sender listing must not call AI')
    result = GroundedRAGChain(model=RunnableLambda(unavailable)).answer_query(db, 'Give all mails from ChatGPT.', user_id)
    assert [c.email_id for c in result['citations']] == [str(emails['personal'].id)]
    assert result['relevance_validated']


def test_partial_validation_returns_citations_with_incomplete_notice(mailbox, monkeypatch):
    import app.rag.chains as chains
    from app.rag.relevance import RelevanceResult
    db, user_id, emails, _ = mailbox
    def partial(docs, *args):
        doc = docs[0]
        doc.metadata['key_points'] = ['Verified planning details.']
        return RelevanceResult(documents=[doc], evaluated=1, failed=True, error_type='timeout')
    monkeypatch.setattr(chains, 'validate_email_relevance', partial)
    result = GroundedRAGChain(model=planning_model()).answer_query(db, 'Find all planning emails', user_id)
    assert result['context_count'] == 1
    assert not result['relevance_validated']
    assert 'Search is incomplete' in result['answer']
    assert 'Verified matches so far' in result['answer']


def test_local_month_boundary_excludes_previous_month_and_next_month(mailbox, monkeypatch):
    from app.rag.query_constraints import mailbox_time_context
    db, user_id, emails, _ = mailbox
    monkeypatch.setattr("app.rag.query_constraints.mailbox_now", lambda: datetime(2026, 10, 19, tzinfo=timezone.utc))
    emails["personal"].received_at = datetime(2026, 9, 30, 18, 30, tzinfo=timezone.utc)
    emails["inbox"].received_at = datetime(2026, 10, 31, 18, 30, tzinfo=timezone.utc)
    db.commit()
    with mailbox_time_context("Asia/Kolkata"):
        ids = primary_email_ids(db, user_id, "Find all interview emails of this month")
    assert ids == [str(emails["personal"].id)]

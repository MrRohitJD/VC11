"""Regression tests for mailbox ownership at RAG boundaries."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from starlette.requests import Request

from app.api.routes.rag import get_session_details, get_user_for_rag
from app.models.email import Base, Email, GmailAccount, User
from app.models.rag import ChatMessage, ChatRetrieval, ChatSession, EmailChunk, EmailEmbedding
from app.rag.indexing import RAGIndexingPipeline
from app.rag.memory import record_retrieval_citations
from app.rag.pgvector_store import PostgresVectorStoreManager
from app.rag.retrieval import RetrievedCitation


class FixedEmbeddings:
    dimensions = 3072
    provider = "test"
    model_name = "test-3072"

    def embed_query(self, _text: str) -> list[float]:
        return [1.0] + [0.0] * (self.dimensions - 1)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


@pytest.fixture()
def db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def make_user(db: Session, label: str) -> tuple[User, GmailAccount]:
    user = User(google_id=f"google-{label}-{uuid4()}", email=f"{label}-{uuid4()}@example.com")
    db.add(user)
    db.flush()
    account = GmailAccount(user_id=user.id, email=user.email, access_token="test-token")
    db.add(account)
    db.flush()
    return user, account


def make_email(db: Session, account: GmailAccount, message_id: str, body: str) -> Email:
    email = Email(
        gmail_account_id=account.id,
        gmail_message_id=message_id,
        subject="Private mailbox message",
        body=body,
        snippet=body,
    )
    db.add(email)
    db.flush()
    return email


def add_embedding(db: Session, email: Email, body: str) -> None:
    chunk = EmailChunk(email_id=email.id, chunk_index=0, chunk_text=body, content_hash="hash")
    db.add(chunk)
    db.flush()
    db.add(EmailEmbedding(
        chunk_id=chunk.id,
        email_id=email.id,
        email_subject=email.subject,
        sender_email=email.from_email or "sender@example.com",
        embedding_text=body,
        chroma_record_id=f"pgvector_{chunk.id}",
        embedding_provider="test",
        embedding_model="test-3072",
        embedding_dimensions=3072,
        embedding=[1.0] + [0.0] * 3071,
        content_hash="hash",
        status="INDEXED",
    ))
    db.flush()


def test_pgvector_search_cannot_cross_user_boundary(db: Session):
    user_a, account_a = make_user(db, "a")
    user_b, account_b = make_user(db, "b")
    email_a = make_email(db, account_a, "message-a", "A secret mailbox message")
    email_b = make_email(db, account_b, "message-b", "B secret mailbox message")
    add_embedding(db, email_a, email_a.body)
    add_embedding(db, email_b, email_b.body)
    db.commit()

    manager = PostgresVectorStoreManager(embedding_function=FixedEmbeddings(), db_session=db)
    b_results = manager.search_with_scores("secret mailbox", user_b.id, top_k=10)

    assert [doc.metadata["email_id"] for doc, _ in b_results] == [str(email_b.id)]
    assert all(doc.metadata["email_id"] != str(email_a.id) for doc, _ in b_results)


def test_indexing_rejects_email_owned_by_another_user(db: Session):
    user_a, account_a = make_user(db, "index-a")
    user_b, _ = make_user(db, "index-b")
    email = make_email(db, account_a, "message-index-a", "A private message")
    db.commit()

    manager = PostgresVectorStoreManager(embedding_function=FixedEmbeddings(), db_session=db)
    pipeline = RAGIndexingPipeline(vector_store_manager=manager)

    with pytest.raises(PermissionError, match="does not belong"):
        pipeline.index_email(db, email.id, user_id=user_b.id)

    assert db.scalars(select(EmailChunk).where(EmailChunk.email_id == email.id)).all() == []


def test_retrieval_audit_drops_foreign_email_citation(db: Session):
    user_a, account_a = make_user(db, "audit-a")
    user_b, _ = make_user(db, "audit-b")
    email_a = make_email(db, account_a, "message-audit-a", "A private message")
    add_embedding(db, email_a, email_a.body)
    chunk_a = db.scalar(select(EmailChunk).where(EmailChunk.email_id == email_a.id))
    session = ChatSession(user_id=user_b.id, title="B session")
    db.add(session)
    db.flush()
    message = ChatMessage(chat_session_id=session.id, role="assistant", content="Answer")
    db.add(message)
    db.commit()

    record_retrieval_citations(
        db,
        session.id,
        message.id,
        [RetrievedCitation(
            email_id=str(email_a.id),
            subject=email_a.subject,
            sender="sender@example.com",
            date="",
            snippet=email_a.body,
            score=1.0,
        )],
    )

    assert db.scalars(select(ChatRetrieval)).all() == []

    # Even if a legacy/corrupt row already exists, the session endpoint must
    # not expose a chunk from another user's mailbox.
    db.add(ChatRetrieval(
        chat_session_id=session.id,
        chat_message_id=message.id,
        chunk_id=chunk_a.id,
        similarity_score=1.0,
        retrieval_rank=1,
    ))
    db.commit()
    details = get_session_details(str(session.id), user_b, db)
    assert details["messages"][0]["citations"] == []


def test_rag_authentication_fails_closed_without_session(db: Session):
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/api/rag/chat",
        "headers": [],
        "query_string": b"",
    })

    with pytest.raises(HTTPException) as error:
        get_user_for_rag(request, db)

    assert error.value.status_code == 401

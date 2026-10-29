"""Unit and integration test suite for LangChain Gmail RAG pipeline.

Tests:
1. Document loading & HTML cleaning
2. LangChain RecursiveCharacterTextSplitter chunking
3. Azure OpenAI Embeddings & Deterministic test fallback
5. Hybrid PostgreSQL retrieval
6. LangChain GroundedRAGChain runnable composition
7. LangChain Tools & Human-in-the-Loop approval gating
8. Query Router & Query Transformation
9. FastAPI RAG endpoints (/api/rag/status, /api/rag/chat, /api/rag/approvals)
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime
from uuid import uuid4
from unittest.mock import patch

from langchain_core.documents import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.email import Base, Email, GmailAccount, User
from app.models.rag import EmailChunk, EmailSendApproval
from app.rag.chains import GroundedRAGChain, format_context_from_citations
from app.rag.chunking import EmailChunker
from app.rag.embeddings import DeterministicOfflineEmbeddings, get_embedding_model
from app.rag.loaders import GmailDocumentAdapter
from app.rag.preprocessing import clean_email_text
from app.rag.retrieval import GmailHybridRetriever, RetrievedCitation
from app.rag.tools import create_gmail_tools


class TestRAGPreprocessingAndChunking(unittest.TestCase):
    """Test document cleaning and LangChain text splitting."""

    def test_clean_email_text_strips_html_and_replies(self):
        html_input = """
        <html>
            <head><style>body { color: red; }</style></head>
            <body>
                <p>Hello team, here is the weekly project update.</p>
                <blockquote>On Mon, Oct 12, 2026, John wrote: previous message</blockquote>
                -- <br>Best regards,<br>Alice
            </body>
        </html>
        """
        cleaned = clean_email_text(html_input)
        self.assertIn("Hello team, here is the weekly project update.", cleaned)
        self.assertNotIn("<style>", cleaned)
        self.assertNotIn("previous message", cleaned)

    def test_chunk_email_short_stays_atomic(self):
        chunker = EmailChunker(chunk_size=500, chunk_overlap=50)
        short_text = "Your flight booking confirmation for SFO to JFK is confirmed."
        chunks = chunker.chunk_email(
            text=short_text,
            email_id=uuid4(),
            user_id=uuid4(),
            subject="Flight Confirmation",
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertIn(short_text, chunks[0].chunk_text)

    def test_chunk_email_large_creates_overlapping_chunks(self):
        chunker = EmailChunker(chunk_size=150, chunk_overlap=30)
        long_text = " ".join([f"Sentence {i} describing project status update in detail." for i in range(25)])
        chunks = chunker.chunk_email(
            text=long_text,
            email_id=uuid4(),
            user_id=uuid4(),
            subject="Long Project Status",
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[1].chunk_index, 1)
        self.assertTrue(chunks[0].content_hash)


class TestToolsAndHumanInTheLoop(unittest.TestCase):
    """Test LangChain tool execution and approval gating."""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.db = self.SessionLocal()

        self.user_id = uuid4()
        self.user = User(
            id=self.user_id,
            google_id="google_test_123",
            email="test@example.com",
            name="Test User",
        )
        self.db.add(self.user)

        self.account = GmailAccount(
            id=uuid4(),
            user_id=self.user_id,
            email="test@example.com",
            access_token="dummy_access_token",
        )
        self.db.add(self.account)
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_human_in_the_loop_draft_and_send_approval(self):
        tools = create_gmail_tools(self.db, self.user_id)
        tool_map = {t.name: t for t in tools}

        draft_tool = tool_map["draft_email"]
        send_tool = tool_map["send_email"]

        # 1. Draft email
        draft_msg = draft_tool.invoke({
            "recipient": "recruiter@example.com",
            "subject": "Thank you for the interview",
            "body": "I appreciate your time today.",
        })
        self.assertIn("Approval ID:", draft_msg)
        self.assertIn("Requires human confirmation", draft_msg)

        # Inspect DB record
        approval = self.db.query(EmailSendApproval).filter_by(user_id=self.user_id).first()
        self.assertIsNotNone(approval)
        self.assertEqual(approval.status, "PENDING")

        # 2. Attempt to send while PENDING -> MUST FAIL (Security Guarantee)
        fail_res = send_tool.invoke({"approval_id": str(approval.id)})
        self.assertIn("Cannot send email. Status is currently 'PENDING'", fail_res)

        # 3. Explicit Human Approval
        approval.status = "APPROVED"
        self.db.commit()

        # 4. Now send -> MUST SUCCEED
        with patch("app.rag.tools.send_gmail_message", return_value={"id": "gmail-message-1"}) as delivery:
            success_res = send_tool.invoke({"approval_id": str(approval.id)})
        delivery.assert_called_once()
        self.assertIn("Email successfully sent", success_res)
        self.assertEqual(approval.status, "SENT")
        self.assertEqual(approval.delivery_message_id, "gmail-message-1")

    def test_schedule_email_requires_approval_and_persists_time(self):
        tools = {t.name: t for t in create_gmail_tools(self.db, self.user_id)}
        scheduled = tools["schedule_email"].invoke({
            "recipient": "recruiter@example.com",
            "subject": "Follow-up",
            "body": "Let us reconnect next week.",
            "scheduled_for": "2099-01-01T10:00:00+00:00",
        })
        self.assertIn("Approval ID:", scheduled)
        approval = self.db.query(EmailSendApproval).filter_by(user_id=self.user_id).first()
        self.assertEqual(approval.status, "PENDING")
        self.assertEqual(approval.scheduled_for.year, 2099)


class TestGroundedRAGChain(unittest.TestCase):
    """Test format context and RAG chain composition."""

    def test_format_context_from_citations(self):
        cits = [
            RetrievedCitation(
                email_id=str(uuid4()),
                subject="Offer Letter",
                sender="hr@company.com",
                date="2026-10-10",
                snippet="We are pleased to offer you the Senior Architect position.",
                score=0.95,
            )
        ]
        context = format_context_from_citations(cits)
        self.assertIn("Offer Letter", context)
        self.assertIn("hr@company.com", context)
        self.assertIn("Senior Architect position", context)


class TestAgentPatterns(unittest.TestCase):
    """Test router, query transformation, compression, and agent patterns."""

    def setUp(self):
        from app.rag.agent import GmailAIAssistant
        self.assistant = GmailAIAssistant()

    def test_router_intent_heuristics(self):
        self.assertEqual(self.assistant.route_intent("Draft an email to recruiter thanking them"), "ACTION_DRAFT")
        self.assertEqual(
            self.assistant.route_intent("Draft a professional mail to adijadhav651@gmail.com related to education load"),
            "ACTION_DRAFT",
        )
        self.assertEqual(self.assistant.route_intent("Send the draft email now"), "ACTION_SEND")
        self.assertEqual(self.assistant.route_intent("Summarize my inbox for today"), "SUMMARIZE")
        self.assertEqual(
            self.assistant.route_intent("Find all interview emails from last week and then extract recruiter details and draft replies"),
            "PLAN_AND_EXECUTE",
        )
        self.assertEqual(self.assistant.route_intent("When is my technical interview scheduled?"), "RAG_QUERY")
        self.assertEqual(
            self.assistant.route_intent("Check for security alerts, 2FA notifications, or password resets."),
            "RAG_QUERY",
        )
        self.assertEqual(
            self.assistant.route_intent("Give me those that are related to chess.com."),
            "RAG_QUERY",
        )
        self.assertEqual(self.assistant.route_intent("What is the capital of France?"), "GENERAL_CHAT")

    def test_deterministic_draft_extraction_requires_explicit_draft_language(self):
        from app.rag.agent import build_deterministic_draft_request

        request = build_deterministic_draft_request(
            "Draft a professional mail to adijadhav651@gmail.com related to education load"
        )
        self.assertEqual(request["recipient"], "adijadhav651@gmail.com")
        self.assertEqual(request["subject"], "Regarding education load")
        self.assertIn("education load", request["body"])
        self.assertIsNone(build_deterministic_draft_request("Search emails from adijadhav651@gmail.com"))

    def test_contextual_compression(self):
        cits = [
            RetrievedCitation(email_id="1", subject="Sub 1", sender="a@b.com", date="2026-10-10", snippet="snip 1", score=0.9),
            RetrievedCitation(email_id="1", subject="Sub 1", sender="a@b.com", date="2026-10-10", snippet="snip 2", score=0.8),
            RetrievedCitation(email_id="2", subject="Sub 2", sender="c@d.com", date="2026-10-11", snippet="snip 3", score=0.1),  # below min
        ]
        compressed = self.assistant.compress_context(cits, min_relevance=0.35)
        self.assertEqual(len(compressed), 2)
        # Low relevance citation 2 was compressed out
        self.assertNotIn("c@d.com", [c.sender for c in compressed])

    def test_query_transformation_fallback(self):
        # Without conversational history, query remains unchanged
        transformed = self.assistant.transform_query("When is the meeting?", [])
        self.assertEqual(transformed, "When is the meeting?")


class TestRAGApiEndpoints(unittest.TestCase):
    """Test FastAPI RAG endpoints."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from app.api.app import app
        from app.api.routes.rag import get_user_for_rag
        from app.database.connection import engine
        from app.models.email import Base
        from app.models.rag import Base as RagBase
        Base.metadata.create_all(bind=engine)
        RagBase.metadata.create_all(bind=engine)
        self.mock_user = User(id=uuid4(), google_id="mock_g", email="mock@example.com", name="Mock User", is_demo=False)
        app.dependency_overrides[get_user_for_rag] = lambda: self.mock_user
        self.client = TestClient(app)

    def tearDown(self):
        from app.api.app import app
        from app.api.routes.rag import get_user_for_rag
        app.dependency_overrides.pop(get_user_for_rag, None)

    def test_rag_status_endpoint(self):
        res = self.client.get("/api/rag/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "operational")
        self.assertTrue("vector_store" in data)
        store = data.get("vector_store")
        self.assertIn("total_vectors", store)

    def test_rag_approvals_endpoint(self):
        res = self.client.get("/api/rag/approvals")
        self.assertEqual(res.status_code, 200)
        self.assertIsInstance(res.json(), list)

    def test_rag_chat_endpoint_empty_query(self):
        res = self.client.post("/api/rag/chat", json={"query": ""})
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()

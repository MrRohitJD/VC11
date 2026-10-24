"""Unit and integration test suite for LangChain Gmail RAG pipeline.

Tests:
1. Document loading & HTML cleaning
2. LangChain RecursiveCharacterTextSplitter chunking
3. Azure OpenAI Embeddings & Deterministic test fallback
4. ChromaDB persistence and user isolation
5. Hybrid retrieval (ChromaDB + PostgreSQL)
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
from app.rag.vectorstore import ChromaVectorStoreManager


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


class TestEmbeddingsAndVectorStore(unittest.TestCase):
    """Test embedding fallback and ChromaDB vector store isolation."""

    def setUp(self):
        self.test_dir = f"./data/test_chroma_{uuid4().hex[:8]}"
        self.embeddings = DeterministicOfflineEmbeddings(dimensions=3072)
        self.vstore_manager = ChromaVectorStoreManager(
            persist_directory=self.test_dir,
            collection_name=f"test_coll_{uuid4().hex[:8]}",
            embedding_model=self.embeddings,
        )

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir, ignore_errors=True)
            except Exception:
                pass

    def test_deterministic_embedding_dimension(self):
        emb = self.embeddings.embed_query("Interview scheduling request")
        self.assertEqual(len(emb), 3072)

    def test_chroma_add_and_user_isolated_search(self):
        user_a = uuid4()
        user_b = uuid4()
        email_a = uuid4()
        email_b = uuid4()

        docs_a = [
            Document(
                page_content="Interview invitation with Google Cloud team on Monday at 10am.",
                metadata={"user_id": str(user_a), "email_id": str(email_a), "subject": "Google Interview", "chunk_index": 0},
            )
        ]
        docs_b = [
            Document(
                page_content="Invoice #98234 for AWS Cloud hosting is due tomorrow.",
                metadata={"user_id": str(user_b), "email_id": str(email_b), "subject": "AWS Invoice", "chunk_index": 0},
            )
        ]

        self.vstore_manager.add_documents(docs_a, [f"{email_a}_0"])
        self.vstore_manager.add_documents(docs_b, [f"{email_b}_0"])

        # Query user A
        results_a = self.vstore_manager.search("interview schedule", user_id=user_a, top_k=5)
        self.assertTrue(len(results_a) >= 1)
        self.assertIn("Google Cloud", results_a[0].page_content)

        # Confirm user B's emails do NOT appear for user A (User Isolation Guarantee)
        for doc in results_a:
            self.assertEqual(doc.metadata["user_id"], str(user_a))


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
        success_res = send_tool.invoke({"approval_id": str(approval.id)})
        self.assertIn("Email successfully sent", success_res)
        self.assertEqual(approval.status, "SENT")


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
        self.assertTrue("vector_store" in data or "chromadb" in data)
        store = data.get("vector_store") or data.get("chromadb")
        self.assertIn("total_vectors", store)

    def test_rag_approvals_endpoint(self):
        res = self.client.get("/api/rag/approvals")
        self.assertEqual(res.status_code, 200)
        self.assertIsInstance(res.json(), list)

    def test_rag_chat_endpoint_empty_query(self):
        res = self.client.post("/api/rag/chat", json={"query": ""})
        self.assertEqual(res.status_code, 400)


class TestLangChainRetrievers(unittest.TestCase):
    """Unit and integration tests for the 4 LangChain retrievers."""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.db = self.SessionLocal()

        self.user_a = uuid4()
        self.user_b = uuid4()

        self.db_user_a = User(id=self.user_a, google_id="g_a", email="a@example.com", name="User A")
        self.db_user_b = User(id=self.user_b, google_id="g_b", email="b@example.com", name="User B")
        self.db.add_all([self.db_user_a, self.db_user_b])

        self.acct_a = GmailAccount(id=uuid4(), user_id=self.user_a, email="a@example.com", access_token="tok_a")
        self.acct_b = GmailAccount(id=uuid4(), user_id=self.user_b, email="b@example.com", access_token="tok_b")
        self.db.add_all([self.acct_a, self.acct_b])

        # Add sample emails for User A
        self.email_1 = Email(
            id=uuid4(),
            gmail_account_id=self.acct_a.id,
            gmail_message_id="msg_google_1",
            from_email="hr@google.com",
            to_email="a@example.com",
            subject="Technical Interview Confirmation",
            body="Your technical interview is scheduled for Friday at 10 AM PST.",
            snippet="Technical interview scheduled for Friday at 10 AM PST.",
            received_at=datetime(2026, 10, 10, 10, 0, 0),
            is_read=True,
            thread_id="thread_google_1",
        )
        self.email_2 = Email(
            id=uuid4(),
            gmail_account_id=self.acct_a.id,
            gmail_message_id="msg_aws_2",
            from_email="billing@aws.amazon.com",
            to_email="a@example.com",
            subject="Monthly Cloud Billing Statement",
            body="Your AWS invoice for October is $45.20 and due by Oct 25.",
            snippet="AWS invoice for October is $45.20.",
            received_at=datetime(2026, 10, 12, 14, 0, 0),
            is_read=False,
            thread_id="thread_aws_2",
        )
        self.db.add_all([self.email_1, self.email_2])
        self.db.commit()

        # Chroma vector store setup with deterministic embeddings
        self.test_dir = f"./data/test_chroma_{uuid4().hex[:8]}"
        self.embeddings = DeterministicOfflineEmbeddings(dimensions=3072)
        self.vstore = ChromaVectorStoreManager(
            persist_directory=self.test_dir,
            collection_name=f"test_retr_{uuid4().hex[:8]}",
            embedding_model=self.embeddings,
        )

        # Add chunks for email 1 and email 2
        docs = [
            Document(
                page_content="Technical interview scheduled for Friday at 10 AM PST with Google Cloud Engineering.",
                metadata={
                    "user_id": str(self.user_a),
                    "email_id": str(self.email_1.id),
                    "subject": self.email_1.subject,
                    "sender": self.email_1.from_email,
                    "received_at": self.email_1.received_at.isoformat(),
                    "thread_id": self.email_1.thread_id,
                    "is_read": self.email_1.is_read,
                },
            ),
            Document(
                page_content="AWS invoice for October is $45.20 and due by Oct 25.",
                metadata={
                    "user_id": str(self.user_a),
                    "email_id": str(self.email_2.id),
                    "subject": self.email_2.subject,
                    "sender": self.email_2.from_email,
                    "received_at": self.email_2.received_at.isoformat(),
                    "thread_id": self.email_2.thread_id,
                    "is_read": self.email_2.is_read,
                },
            ),
        ]
        self.vstore.add_documents(docs)

    def tearDown(self):
        self.db.close()
        import shutil
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir, ignore_errors=True)
            except Exception:
                pass

    def test_chroma_semantic_retriever(self):
        from app.rag.retrieval import ChromaSemanticRetriever

        retriever = ChromaSemanticRetriever(
            user_id=str(self.user_a),
            top_k=2,
            chroma_manager=self.vstore,
        )
        docs = retriever.invoke("interview confirmation")
        self.assertTrue(len(docs) >= 1)
        self.assertIn("interview", docs[0].page_content.lower())
        self.assertIn("score", docs[0].metadata)

        # Tenant isolation: User B should get 0 results
        retriever_b = ChromaSemanticRetriever(
            user_id=str(self.user_b),
            top_k=2,
            chroma_manager=self.vstore,
        )
        docs_b = retriever_b.invoke("interview confirmation")
        self.assertEqual(len(docs_b), 0)

    def test_gmail_custom_hybrid_retriever_filters(self):
        from app.rag.retrieval import GmailCustomHybridRetriever

        # Test sender filter
        retriever_sender = GmailCustomHybridRetriever(
            user_id=str(self.user_a),
            db_session=self.db,
            sender="google.com",
            chroma_manager=self.vstore,
        )
        docs_sender = retriever_sender.invoke("interview")
        self.assertTrue(len(docs_sender) >= 1)
        self.assertEqual(docs_sender[0].metadata["sender"], "hr@google.com")

        # Test date range filter
        retriever_date = GmailCustomHybridRetriever(
            user_id=str(self.user_a),
            db_session=self.db,
            start_date=datetime(2026, 10, 11),
            chroma_manager=self.vstore,
        )
        docs_date = retriever_date.invoke("statement")
        self.assertTrue(len(docs_date) >= 1)
        self.assertEqual(docs_date[0].metadata["email_id"], str(self.email_2.id))

        # Test read status filter
        retriever_unread = GmailCustomHybridRetriever(
            user_id=str(self.user_a),
            db_session=self.db,
            is_read=False,
            chroma_manager=self.vstore,
        )
        docs_unread = retriever_unread.invoke("AWS invoice")
        self.assertTrue(len(docs_unread) >= 1)
        self.assertFalse(docs_unread[0].metadata["is_read"])

        # Test thread_id filter
        retriever_thread = GmailCustomHybridRetriever(
            user_id=str(self.user_a),
            db_session=self.db,
            thread_id="thread_google_1",
            chroma_manager=self.vstore,
        )
        docs_thread = retriever_thread.invoke("interview")
        self.assertTrue(len(docs_thread) >= 1)
        self.assertEqual(docs_thread[0].metadata["thread_id"], "thread_google_1")

        # Test email_id filter
        retriever_eid = GmailCustomHybridRetriever(
            user_id=str(self.user_a),
            db_session=self.db,
            email_id=str(self.email_1.id),
            chroma_manager=self.vstore,
        )
        docs_eid = retriever_eid.invoke("anything")
        self.assertTrue(len(docs_eid) >= 1)
        self.assertEqual(docs_eid[0].metadata["email_id"], str(self.email_1.id))

    def test_history_aware_retriever(self):
        from app.rag.retrieval import ChromaSemanticRetriever, HistoryAwareRetriever
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langchain_core.messages import HumanMessage, AIMessage

        base = ChromaSemanticRetriever(
            user_id=str(self.user_a),
            top_k=2,
            chroma_manager=self.vstore,
        )

        history = [
            HumanMessage(content="Did I receive an email about my Google interview?"),
            AIMessage(content="Yes, you received a confirmation for Friday at 10 AM PST."),
        ]

        # 1. Test LCEL prompt | llm | StrOutputParser chain with simulated LLM
        mock_llm = FakeListChatModel(responses=["When is the Google technical interview scheduled?"])
        history_aware_llm = HistoryAwareRetriever(
            base_retriever=base,
            chat_history=history,
            llm=mock_llm,
        )
        docs_llm = history_aware_llm.invoke("When is it scheduled?")
        self.assertTrue(len(docs_llm) >= 1)
        self.assertIn("interview", docs_llm[0].page_content.lower())
        self.assertEqual(docs_llm[0].metadata.get("rewritten_query"), "When is the Google technical interview scheduled?")

        # 2. Test heuristic fallback when LLM is None
        history_aware_fallback = HistoryAwareRetriever(
            base_retriever=base,
            chat_history=history,
        )
        docs_fb = history_aware_fallback.invoke("When is it scheduled?")
        self.assertTrue(len(docs_fb) >= 1)
        self.assertIn("interview", docs_fb[0].page_content.lower())

    def test_multi_query_retriever(self):
        from app.rag.retrieval import ChromaSemanticRetriever, MultiQueryRetriever

        base = ChromaSemanticRetriever(
            user_id=str(self.user_a),
            top_k=2,
            chroma_manager=self.vstore,
        )

        mq_retriever = MultiQueryRetriever(
            base_retriever=base,
            query_count=3,
        )

        docs = mq_retriever.invoke("cloud invoice payment")
        self.assertTrue(len(docs) >= 1)
        # Deduplication check: every returned document has unique email_id
        seen_ids = set()
        for d in docs:
            eid = d.metadata.get("email_id")
            self.assertNotIn(eid, seen_ids)
            seen_ids.add(eid)


if __name__ == "__main__":
    unittest.main()

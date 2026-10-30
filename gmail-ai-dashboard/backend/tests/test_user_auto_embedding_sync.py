"""Tests for user login/signup auto-embedding and complete PG vector population."""

from datetime import datetime, timezone
import unittest
from uuid import uuid4

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker

from app.models.email import Base, Email, GmailAccount, User
from app.models.rag import Base as RagBase, EmailChunk, EmailEmbedding
from app.rag.embeddings import DeterministicOfflineEmbeddings
from app.services.rag_indexer import default_rag_service


class TestUserAutoEmbeddingSync(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        RagBase.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

        # Use deterministic offline embeddings for tests
        self._orig_embed_fn = default_rag_service.pipeline.embedding_function
        default_rag_service.pipeline.embedding_function = DeterministicOfflineEmbeddings(dimensions=3072)

        self.user = User(
            id=uuid4(),
            google_id="user_google_123",
            email="testuser@example.com",
            name="Test User",
            is_demo=False,
        )
        self.db.add(self.user)
        self.db.flush()

        self.account = GmailAccount(
            id=uuid4(),
            user_id=self.user.id,
            email="testuser@example.com",
            access_token="fake_token",
            sync_status="IDLE",
        )
        self.db.add(self.account)
        self.db.commit()

    def tearDown(self):
        default_rag_service.pipeline.embedding_function = self._orig_embed_fn
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        RagBase.metadata.drop_all(bind=self.engine)

    def test_auto_embedding_fills_pgvector_table_with_metadata(self):
        """All user emails (including snippet-only and subject-only) are embedded with metadata."""
        email1 = Email(
            id=uuid4(),
            gmail_account_id=self.account.id,
            gmail_message_id="msg_001",
            subject="Quarterly Budget Planning",
            body="The budget planning meeting will take place on Friday afternoon.",
            from_email="finance@example.com",
            to_email="testuser@example.com",
            received_at=datetime.now(timezone.utc),
        )
        email2 = Email(
            id=uuid4(),
            gmail_account_id=self.account.id,
            gmail_message_id="msg_002",
            subject="Quick Reminder",
            body="",  # Empty body, has snippet
            snippet="Don't forget to submit your weekly timesheet by 5pm.",
            from_email="hr@example.com",
            to_email="testuser@example.com",
            received_at=datetime.now(timezone.utc),
        )
        email3 = Email(
            id=uuid4(),
            gmail_account_id=self.account.id,
            gmail_message_id="msg_003",
            subject="Calendar invite: Sync discussion",
            body=None,  # No body and no snippet, only subject
            snippet=None,
            from_email="boss@example.com",
            to_email="testuser@example.com",
            received_at=datetime.now(timezone.utc),
        )
        self.db.add_all([email1, email2, email3])
        self.db.commit()

        # Before indexing: 3 unindexed emails
        unindexed = default_rag_service.get_unindexed_email_ids(self.db, user_id=self.user.id)
        self.assertEqual(len(unindexed), 3)

        # Execute indexing
        results = default_rag_service.index_unindexed_emails(self.db, user_id=self.user.id)
        self.assertEqual(results["emails_indexed"], 3)
        self.assertEqual(results["chunks_created"], 3)

        # After indexing: 0 unindexed emails remain
        unindexed_after = default_rag_service.get_unindexed_email_ids(self.db, user_id=self.user.id)
        self.assertEqual(len(unindexed_after), 0)

        # Verify pgvector table email_embeddings is completely populated with metadata
        embeddings = list(self.db.scalars(
            select(EmailEmbedding).join(EmailChunk, EmailChunk.id == EmailEmbedding.chunk_id)
            .where(EmailChunk.email_id.in_([email1.id, email2.id, email3.id]))
        ).all())
        self.assertEqual(len(embeddings), 3)

        for emb in embeddings:
            self.assertEqual(emb.status, "INDEXED")
            self.assertIsNotNone(emb.embedding)
            self.assertEqual(len(emb.embedding), 3072)
            self.assertIsNotNone(emb.email_subject)
            self.assertIsNotNone(emb.sender_email)
            self.assertIsNotNone(emb.embedding_text)

    def test_purge_unembedded_emails_ensures_no_unembedded_email_remains(self):
        """Any email that has not been embedded must not remain in the database."""
        email_valid = Email(
            id=uuid4(),
            gmail_account_id=self.account.id,
            gmail_message_id="msg_valid",
            subject="Valid message",
            body="This email will be embedded normally.",
            from_email="alice@example.com",
            to_email="testuser@example.com",
            received_at=datetime.now(timezone.utc),
        )
        email_unembedded = Email(
            id=uuid4(),
            gmail_account_id=self.account.id,
            gmail_message_id="msg_unembedded",
            subject="Orphan message",
            body="This email will remain unembedded.",
            from_email="bob@example.com",
            to_email="testuser@example.com",
            received_at=datetime.now(timezone.utc),
        )
        self.db.add_all([email_valid, email_unembedded])
        self.db.commit()

        # Index only email_valid
        default_rag_service.index_email(self.db, email_valid.id, user_id=self.user.id)

        # Purge unembedded emails
        purged = default_rag_service.purge_unembedded_emails(self.db, user_id=self.user.id)
        self.assertEqual(purged, 1)

        # Check emails table: email_unembedded is removed, email_valid remains
        remaining_emails = list(self.db.scalars(
            select(Email).where(Email.gmail_account_id == self.account.id)
        ).all())
        self.assertEqual(len(remaining_emails), 1)
        self.assertEqual(remaining_emails[0].id, email_valid.id)

    def test_purge_unembedded_emails_protects_demo_account(self):
        """Demo account emails without embeddings should not be purged during sweeps."""
        demo_user = User(
            id=uuid4(),
            google_id="demo_google_id",
            email="demo@gmail.com",
            name="Demo Account",
            is_demo=True,
        )
        self.db.add(demo_user)
        self.db.flush()

        demo_account = GmailAccount(
            id=uuid4(),
            user_id=demo_user.id,
            email="demo@gmail.com",
            access_token="fake_token",
            sync_status="IDLE",
        )
        self.db.add(demo_account)
        self.db.flush()

        demo_email = Email(
            id=uuid4(),
            gmail_account_id=demo_account.id,
            gmail_message_id="msg_demo_01",
            subject="Demo Welcome",
            body="Welcome to the demo mailbox.",
            from_email="system@example.com",
            to_email="demo@gmail.com",
            received_at=datetime.now(timezone.utc),
        )
        self.db.add(demo_email)
        self.db.commit()

        # Run global purge (user_id=None)
        default_rag_service.purge_unembedded_emails(self.db, user_id=None)

        # Demo email must NOT be purged
        demo_emails_after = list(self.db.scalars(
            select(Email).where(Email.gmail_account_id == demo_account.id)
        ).all())
        self.assertEqual(len(demo_emails_after), 1)
        self.assertEqual(demo_emails_after[0].id, demo_email.id)

    def test_bulk_embedding_processes_multiple_emails_in_single_call(self):
        """Bulk embedding groups chunk texts from multiple emails into a single embed_documents call."""
        emails = [
            Email(
                id=uuid4(),
                gmail_account_id=self.account.id,
                gmail_message_id=f"bulk_msg_{i}",
                subject=f"Bulk Subject {i}",
                body=f"Bulk email message body {i} containing important work details.",
                from_email=f"sender{i}@example.com",
                to_email="testuser@example.com",
                received_at=datetime.now(timezone.utc),
            )
            for i in range(5)
        ]
        self.db.add_all(emails)
        self.db.commit()

        call_batches = []
        original_embed = default_rag_service.pipeline.embedding_function.embed_documents

        def spy_embed(texts):
            call_batches.append(list(texts))
            return original_embed(texts)

        default_rag_service.pipeline.embedding_function.embed_documents = spy_embed
        try:
            res = default_rag_service.index_unindexed_emails(self.db, user_id=self.user.id)
            self.assertEqual(res["emails_indexed"], 5)
            self.assertEqual(res["chunks_created"], 5)

            # Assert that embed_documents was called only ONCE for all 5 emails instead of 5 times!
            self.assertEqual(len(call_batches), 1)
            self.assertEqual(len(call_batches[0]), 5)

            # Verify all 5 embeddings are saved
            for email in emails:
                emb = self.db.scalar(
                    select(EmailEmbedding).join(EmailChunk, EmailChunk.id == EmailEmbedding.chunk_id)
                    .where(EmailChunk.email_id == email.id)
                )
                self.assertIsNotNone(emb)
                self.assertEqual(emb.status, "INDEXED")
        finally:
            default_rag_service.pipeline.embedding_function.embed_documents = original_embed

    def test_bulk_embedding_fallback_on_batch_failure(self):
        """When a bulk batch fails, it falls back to individual email indexing without failing all."""
        emails = [
            Email(
                id=uuid4(),
                gmail_account_id=self.account.id,
                gmail_message_id=f"fallback_msg_{i}",
                subject=f"Fallback Subject {i}",
                body=f"Body {i}",
                from_email=f"user{i}@example.com",
                to_email="testuser@example.com",
                received_at=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]
        self.db.add_all(emails)
        self.db.commit()

        original_embed = default_rag_service.pipeline.embedding_function.embed_documents
        first_call = True

        def failing_bulk_embed(texts):
            nonlocal first_call
            if first_call and len(texts) > 1:
                first_call = False
                raise RuntimeError("Simulated bulk API failure")
            return original_embed(texts)

        default_rag_service.pipeline.embedding_function.embed_documents = failing_bulk_embed
        try:
            res = default_rag_service.index_unindexed_emails(self.db, user_id=self.user.id)
            # The fallback successfully indexed all emails individually!
            self.assertEqual(res["emails_indexed"], 3)
            self.assertEqual(res["chunks_created"], 3)
        finally:
            default_rag_service.pipeline.embedding_function.embed_documents = original_embed


if __name__ == "__main__":
    unittest.main()

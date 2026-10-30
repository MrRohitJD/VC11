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


if __name__ == "__main__":
    unittest.main()

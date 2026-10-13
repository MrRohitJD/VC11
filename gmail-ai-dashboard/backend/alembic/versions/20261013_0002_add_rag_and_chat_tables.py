"""Add RAG chunking, embeddings mapping, attachments, chat sessions, and email send approvals.

Revision ID: 20261013_0002
Revises: 20261012_0001
Create Date: 2026-10-13 22:20:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20261013_0002"
down_revision = "20261012_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. attachments
    op.create_table(
        "attachments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email_id", sa.Uuid(), nullable=False),
        sa.Column("gmail_attachment_id", sa.String(length=255), nullable=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=150), nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("storage_uri", sa.Text(), nullable=True),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("extraction_status", sa.String(length=30), server_default="PENDING", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.CheckConstraint(
            "extraction_status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'SKIPPED')",
            name="ck_attachments_extraction_status",
        ),
        sa.ForeignKeyConstraint(["email_id"], ["emails.id"], name="fk_attachments_email_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_attachments"),
        sa.UniqueConstraint("email_id", "gmail_attachment_id", name="uq_attachments_email_attachment_id"),
    )
    op.create_index("ix_attachments_email_id", "attachments", ["email_id"], unique=False)
    op.create_index("ix_attachments_extraction_status", "attachments", ["extraction_status"], unique=False)

    # 2. email_chunks
    op.create_table(
        "email_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("start_index", sa.Integer(), nullable=True),
        sa.Column("end_index", sa.Integer(), nullable=True),
        sa.Column("chunk_strategy", sa.String(length=50), server_default="recursive_character", nullable=False),
        sa.Column("content_hash", sa.String(length=64), server_default="", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["email_id"], ["emails.id"], name="fk_email_chunks_email_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_email_chunks"),
        sa.UniqueConstraint("email_id", "chunk_index", name="uq_email_chunks_email_index"),
    )
    op.create_index("ix_email_chunks_email_id", "email_chunks", ["email_id"], unique=False)
    op.create_index("ix_email_chunks_content_hash", "email_chunks", ["content_hash"], unique=False)

    # 3. email_embeddings
    op.create_table(
        "email_embeddings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("chroma_collection", sa.String(length=100), server_default="gmail_email_chunks", nullable=False),
        sa.Column("chroma_record_id", sa.String(length=255), nullable=False),
        sa.Column("embedding_provider", sa.String(length=50), server_default="sentence-transformers", nullable=False),
        sa.Column("embedding_model", sa.String(length=100), server_default="all-MiniLM-L6-v2", nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), server_default="384", nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="INDEXED", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.CheckConstraint("status IN ('PENDING', 'INDEXED', 'FAILED', 'OUTDATED')", name="ck_email_embeddings_status"),
        sa.ForeignKeyConstraint(["chunk_id"], ["email_chunks.id"], name="fk_email_embeddings_chunk_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_email_embeddings"),
        sa.UniqueConstraint("chroma_collection", "chroma_record_id", name="uq_email_embeddings_collection_record"),
        sa.UniqueConstraint("chunk_id", "chroma_collection", name="uq_email_embeddings_chunk_collection"),
    )
    op.create_index("ix_email_embeddings_chunk_id", "email_embeddings", ["chunk_id"], unique=False)
    op.create_index("ix_email_embeddings_status", "email_embeddings", ["status"], unique=False)
    op.create_index("ix_email_embeddings_content_hash", "email_embeddings", ["content_hash"], unique=False)

    # 4. chat_sessions
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_chat_sessions_user_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_chat_sessions"),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"], unique=False)
    op.create_index("ix_chat_sessions_updated_at", "chat_sessions", ["updated_at"], unique=False)

    # 5. chat_messages
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("chat_session_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.CheckConstraint("role IN ('user', 'assistant', 'system')", name="ck_chat_messages_role"),
        sa.ForeignKeyConstraint(["chat_session_id"], ["chat_sessions.id"], name="fk_chat_messages_chat_session_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_chat_messages"),
    )
    op.create_index("ix_chat_messages_session_created", "chat_messages", ["chat_session_id", "created_at"], unique=False)

    # 6. chat_retrievals
    op.create_table(
        "chat_retrievals",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("chat_session_id", sa.Uuid(), nullable=False),
        sa.Column("chat_message_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("similarity_score", sa.Float(), nullable=True),
        sa.Column("retrieval_rank", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["chat_session_id"], ["chat_sessions.id"], name="fk_chat_retrievals_chat_session_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chat_message_id"], ["chat_messages.id"], name="fk_chat_retrievals_chat_message_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id"], ["email_chunks.id"], name="fk_chat_retrievals_chunk_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_chat_retrievals"),
        sa.UniqueConstraint("chat_message_id", "chunk_id", name="uq_chat_retrievals_message_chunk"),
    )
    op.create_index("ix_chat_retrievals_message_id", "chat_retrievals", ["chat_message_id"], unique=False)
    op.create_index("ix_chat_retrievals_chunk_id", "chat_retrievals", ["chunk_id"], unique=False)
    op.create_index("ix_chat_retrievals_session_id", "chat_retrievals", ["chat_session_id"], unique=False)

    # 7. email_send_approvals
    op.create_table(
        "email_send_approvals",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("recipient_email", sa.String(length=320), nullable=False),
        sa.Column("subject", sa.Text(), nullable=True),
        sa.Column("draft_body", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="PENDING", nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.CheckConstraint(
            "status IN ('PENDING', 'APPROVED', 'REJECTED', 'SENT', 'CANCELLED')",
            name="ck_email_send_approvals_status",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_email_send_approvals_user_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_email_send_approvals"),
    )
    op.create_index("ix_email_send_approvals_user_status", "email_send_approvals", ["user_id", "status"], unique=False)


def downgrade() -> None:
    # Drop tables strictly in reverse dependency order
    op.drop_index("ix_email_send_approvals_user_status", table_name="email_send_approvals")
    op.drop_table("email_send_approvals")

    op.drop_index("ix_chat_retrievals_session_id", table_name="chat_retrievals")
    op.drop_index("ix_chat_retrievals_chunk_id", table_name="chat_retrievals")
    op.drop_index("ix_chat_retrievals_message_id", table_name="chat_retrievals")
    op.drop_table("chat_retrievals")

    op.drop_index("ix_chat_messages_session_created", table_name="chat_messages")
    op.drop_table("chat_messages")

    op.drop_index("ix_chat_sessions_updated_at", table_name="chat_sessions")
    op.drop_index("ix_chat_sessions_user_id", table_name="chat_sessions")
    op.drop_table("chat_sessions")

    op.drop_index("ix_email_embeddings_content_hash", table_name="email_embeddings")
    op.drop_index("ix_email_embeddings_status", table_name="email_embeddings")
    op.drop_index("ix_email_embeddings_chunk_id", table_name="email_embeddings")
    op.drop_table("email_embeddings")

    op.drop_index("ix_email_chunks_content_hash", table_name="email_chunks")
    op.drop_index("ix_email_chunks_email_id", table_name="email_chunks")
    op.drop_table("email_chunks")

    op.drop_index("ix_attachments_extraction_status", table_name="attachments")
    op.drop_index("ix_attachments_email_id", table_name="attachments")
    op.drop_table("attachments")

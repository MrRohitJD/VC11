from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.models.chat_quota import ChatQuota
from app.models.llm import LlmClassificationJob
from app.models.email import (
	Email,
	EmailClassification,
	EmailLabel,
	GmailAccount,
	SyncHistory,
	User,
)
from app.models.rag import (
	Attachment,
	ChatMessage,
	ChatRetrieval,
	ChatSession,
	EmailChunk,
	EmailEmbedding,
	EmailSendApproval,
)


def upsert_google_user(
	db: Session,
	*,
	google_id: str,
	email: str,
	name: str | None,
	picture: str | None,
	access_token: str,
	refresh_token: str | None,
	token_expiry: datetime | None,
	scope: str | None,
) -> User:
	"""Persist the Google identity and its Gmail OAuth credentials atomically."""
	user = db.scalar(select(User).where(User.google_id == google_id))
	if user is None:
		user = db.scalar(select(User).where(User.email == email))
	if user is not None and user.is_demo:
		raise ValueError("The public demo account cannot be linked to Google")
	if user is None:
		user = User(google_id=google_id, email=email)
		db.add(user)
	else:
		user.google_id = google_id

	user.email = email
	user.name = name
	user.picture = picture
	user.updated_at = datetime.now(timezone.utc)
	db.flush()

	account = db.scalar(select(GmailAccount).where(GmailAccount.user_id == user.id))
	if account is None:
		account = db.scalar(select(GmailAccount).where(GmailAccount.email == email))
	if account is None:
		account = GmailAccount(user_id=user.id, email=email, access_token=access_token)
		db.add(account)
	else:
		account.user_id = user.id
		account.email = email
		account.access_token = access_token

	account.refresh_token = refresh_token or account.refresh_token
	account.token_expiry = token_expiry
	account.scope = scope
	account.updated_at = datetime.now(timezone.utc)
	db.commit()
	db.refresh(user)
	return user


def serialize_user(user: User) -> dict[str, Any]:
	return {
		"id": str(user.id),
		"google_id": user.google_id,
		"email": user.email,
		"name": user.name,
		"picture": user.picture,
		"is_demo": user.is_demo,
		"features": {
			"ai_assistant_enabled": True,
			"send_mail_enabled": not user.is_demo,
		},
	}


def delete_user_gmail_data(db: Session, user: User) -> None:
	"""Erase a user's stored mailbox data while preserving the user and labels.

	Labels are owned by ``gmail_accounts`` in the schema, so the account rows
	must remain as structural parents for the labels.  The account's OAuth and
	sync state is cleared, while emails and every record that depends on an
	email are deleted.  User-owned assistant history and pending send approvals
	are also removed; the ``users`` row itself is intentionally untouched.
	"""
	account_ids = list(db.scalars(
		select(GmailAccount.id).where(GmailAccount.user_id == user.id)
	).all())
	if account_ids:
		# Delete in dependency order with set-based statements.  The previous
		# relationship walk loaded every message, chunk, and attachment into the
		# identity map, making erase slower as the mailbox grew.  These predicates
		# preserve labels while removing every record that contains mail data.
		email_ids = select(Email.id).where(Email.gmail_account_id.in_(account_ids))
		chunk_ids = select(EmailChunk.id).where(EmailChunk.email_id.in_(email_ids))
		db.execute(delete(ChatRetrieval).where(ChatRetrieval.chunk_id.in_(chunk_ids)))
		db.execute(delete(EmailEmbedding).where(
			(EmailEmbedding.email_id.in_(email_ids)) | EmailEmbedding.chunk_id.in_(chunk_ids)
		))
		db.execute(delete(Attachment).where(Attachment.email_id.in_(email_ids)))
		db.execute(delete(EmailClassification).where(EmailClassification.email_id.in_(email_ids)))
		db.execute(delete(EmailLabel).where(EmailLabel.gmail_account_id.in_(account_ids)))
		db.execute(delete(EmailChunk).where(EmailChunk.email_id.in_(email_ids)))
		db.execute(delete(LlmClassificationJob).where(LlmClassificationJob.gmail_account_id.in_(account_ids)))
		db.execute(delete(SyncHistory).where(SyncHistory.gmail_account_id.in_(account_ids)))
		db.execute(delete(Email).where(Email.gmail_account_id.in_(account_ids)))

		# Keep accounts as the structural parents of labels, but remove
		# credentials and mailbox cursor data so no OAuth secret survives.
		db.execute(update(GmailAccount).where(GmailAccount.id.in_(account_ids)).values(
			access_token="", refresh_token=None, token_expiry=None, scope=None,
			gmail_user_id=None, sync_status="IDLE", last_synced_at=None, history_id=None,
		))

	# Clear assistant state as well, without loading message bodies.  This keeps
	# the user row and Gmail labels intact while removing local conversation data.
	session_ids = select(ChatSession.id).where(ChatSession.user_id == user.id)
	db.execute(delete(ChatRetrieval).where(ChatRetrieval.chat_session_id.in_(session_ids)))
	db.execute(delete(ChatMessage).where(ChatMessage.chat_session_id.in_(session_ids)))
	db.execute(delete(ChatSession).where(ChatSession.user_id == user.id))
	db.execute(delete(EmailSendApproval).where(EmailSendApproval.user_id == user.id))
	db.execute(delete(ChatQuota).where(ChatQuota.identity == f"user:{user.id}"))
	db.commit()


def delete_user_and_all_data(db: Session, user: User) -> None:
	"""Delete the user and all associated data (cascades handle related records)."""
	db.delete(user)
	db.commit()

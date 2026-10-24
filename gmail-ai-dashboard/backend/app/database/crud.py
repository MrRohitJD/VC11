from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chat_quota import ChatQuota
from app.models.llm import LlmClassificationJob
from app.models.email import GmailAccount, User
from app.models.rag import ChatSession, EmailSendApproval


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
	}


def delete_user_gmail_data(db: Session, user: User) -> None:
	"""Erase a user's stored mailbox data while preserving the user and labels.

	Labels are owned by ``gmail_accounts`` in the schema, so the account rows
	must remain as structural parents for the labels.  The account's OAuth and
	sync state is cleared, while emails and every record that depends on an
	email are deleted.  User-owned assistant history and pending send approvals
	are also removed; the ``users`` row itself is intentionally untouched.
	"""
	accounts = db.scalars(
		select(GmailAccount).where(GmailAccount.user_id == user.id)
	).all()
	for account in accounts:
		# These jobs can outlive an email while it is being classified, so they
		# must be deleted explicitly before the account is reused.
		for job in list(account.classification_jobs):
			db.delete(job)
		for history in list(account.sync_history):
			db.delete(history)
		for email in list(account.emails):
			db.delete(email)

		# Keep the account as the parent of the labels, but remove credentials
		# and mailbox cursor data so the erase does not retain usable OAuth data.
		account.access_token = ""
		account.refresh_token = None
		account.token_expiry = None
		account.scope = None
		account.gmail_user_id = None
		account.sync_status = "IDLE"
		account.last_synced_at = None
		account.history_id = None

	for session in db.scalars(
		select(ChatSession).where(ChatSession.user_id == user.id)
	).all():
		db.delete(session)
	for approval in db.scalars(
		select(EmailSendApproval).where(EmailSendApproval.user_id == user.id)
	).all():
		db.delete(approval)
	for quota in db.scalars(
		select(ChatQuota).where(ChatQuota.identity == f"user:{user.id}")
	).all():
		db.delete(quota)
	db.commit()


def delete_user_and_all_data(db: Session, user: User) -> None:
	"""Delete the user and all associated data (cascades handle related records)."""
	db.delete(user)
	db.commit()

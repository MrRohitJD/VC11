from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.email import GmailAccount, User


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
	}


def delete_user_gmail_data(db: Session, user: User) -> None:
	"""Delete Gmail accounts and their related stored data, preserving the user."""
	accounts = db.scalars(
		select(GmailAccount).where(GmailAccount.user_id == user.id)
	).all()
	for account in accounts:
		db.delete(account)
	db.commit()


def delete_user_and_all_data(db: Session, user: User) -> None:
	"""Delete the user and all associated data (cascades handle related records)."""
	db.delete(user)
	db.commit()

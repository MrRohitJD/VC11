"""Email & label query routes — all endpoints enforce user-level authorization."""

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.connection import get_db
from app.models.email import Email, EmailLabel, GmailAccount, Label, User

router = APIRouter(prefix="/api", tags=["emails"])


def _get_account(db: Session, user: User) -> GmailAccount:
	account = db.scalar(
		select(GmailAccount).where(GmailAccount.user_id == user.id)
	)
	if account is None:
		raise HTTPException(status_code=404, detail="No Gmail account linked")
	return account


@router.get("/emails")
def list_emails(
	page: int = Query(1, ge=1),
	per_page: int = Query(50, ge=1, le=200),
	search: Optional[str] = Query(None),
	label: Optional[str] = Query(None),
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""List emails for the authenticated user with pagination, search and label filtering."""
	account = _get_account(db, user)

	query = select(Email).where(Email.gmail_account_id == account.id)

	if search:
		pattern = f"%{search}%"
		query = query.where(
			Email.subject.ilike(pattern)
			| Email.from_email.ilike(pattern)
			| Email.snippet.ilike(pattern)
		)

	if label:
		query = query.join(EmailLabel).join(Label).where(Label.name.ilike(label))

	# Total count
	count_query = select(func.count()).select_from(query.subquery())
	total = db.scalar(count_query) or 0

	# Paginated results
	emails = db.scalars(
		query.order_by(Email.received_at.desc().nullslast())
		.offset((page - 1) * per_page)
		.limit(per_page)
	).all()

	return {
		"total": total,
		"page": page,
		"per_page": per_page,
		"emails": [_serialize_email(e) for e in emails],
	}


@router.get("/emails/{email_id}")
def get_email(
	email_id: str,
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""Get a single email by its database UUID. Enforces user ownership."""
	account = _get_account(db, user)
	email = db.scalar(
		select(Email).where(
			Email.id == email_id,
			Email.gmail_account_id == account.id,
		)
	)
	if email is None:
		raise HTTPException(status_code=404, detail="Email not found")
	return _serialize_email(email, include_body=True)


@router.get("/labels")
def list_labels(
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""List all Gmail labels for the authenticated user."""
	account = _get_account(db, user)
	labels = db.scalars(
		select(Label).where(Label.gmail_account_id == account.id).order_by(Label.name)
	).all()
	return {
		"labels": [
			{
				"id": str(l.id),
				"label_id": l.label_id,
				"name": l.name,
				"type": l.type,
			}
			for l in labels
		]
	}


def _serialize_email(email: Email, include_body: bool = False) -> dict:
	result = {
		"id": str(email.id),
		"gmail_message_id": email.gmail_message_id,
		"thread_id": email.thread_id,
		"subject": email.subject,
		"snippet": email.snippet,
		"from_email": email.from_email,
		"to_email": email.to_email,
		"cc": email.cc,
		"bcc": email.bcc,
		"received_at": email.received_at.isoformat() if email.received_at else None,
		"is_read": email.is_read,
		"has_attachment": email.has_attachment,
		"size_estimate": email.size_estimate,
		"labels": [
			{"id": str(el.label.id), "name": el.label.name}
			for el in email.labels
			if el.label
		],
	}
	if include_body:
		result["body"] = email.body
		result["attachments_meta"] = (
			json.loads(email.attachments_meta) if email.attachments_meta else []
		)
	return result

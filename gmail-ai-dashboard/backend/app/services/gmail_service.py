from datetime import datetime
from email.utils import parsedate_to_datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.email import Email, EmailLabel, GmailAccount, Label, User


def store_emails(
	db: Session,
	gmail_account_id: UUID,
	email_records: list[dict],
	service=None,
) -> list[Email]:
	"""Insert fetched Gmail records, updating existing message IDs safely."""
	stored = []

	for record in email_records:
		email = db.scalar(
			select(Email).where(Email.gmail_message_id == record["id"])
		)
		if email is None:
			email = Email(
				gmail_account_id=gmail_account_id,
				gmail_message_id=record["id"],
			)
			db.add(email)

		email.subject = record.get("subject")
		email.body = record.get("body")
		email.from_email = record.get("from")
		email.received_at = _parse_received_at(record.get("date"))
		db.flush()
		_store_email_labels(
			db=db,
			gmail_account_id=gmail_account_id,
			email=email,
			label_ids=record.get("labelIds", []),
			service=service,
		)
		stored.append(email)

	db.commit()
	for email in stored:
		db.refresh(email)
	return stored


def _store_email_labels(
	db: Session,
	gmail_account_id: UUID,
	email: Email,
	label_ids: list[str],
	service,
) -> None:
	if service is None:
		return

	for label_id in label_ids:
		label = db.scalar(
			select(Label).where(
				Label.gmail_account_id == gmail_account_id,
				Label.label_id == label_id,
			)
		)
		if label is None:
			label_data = service.users().labels().get(
				userId="me",
				id=label_id,
			).execute()
			label = Label(
				gmail_account_id=gmail_account_id,
				label_id=label_id,
				name=label_data.get("name", label_id),
				type=label_data.get("type"),
			)
			db.add(label)
			db.flush()

		association = db.scalar(
			select(EmailLabel).where(
				EmailLabel.email_id == email.id,
				EmailLabel.label_id == label.id,
			)
		)
		if association is None:
			db.add(EmailLabel(email_id=email.id, label_id=label.id))


def _parse_received_at(value: str | None) -> datetime | None:
	if not value:
		return None
	try:
		return parsedate_to_datetime(value)
	except (TypeError, ValueError):
		return None


def fetch_and_store_emails(
	db: Session,
	gmail_account_id: UUID,
	count: int = 1,
	start_date: str | None = None,
	end_date: str | None = None,
	service=None,
) -> list[Email]:
	"""Fetch and persist Gmail records, returning clean dictionaries."""
	from app.services.get_mail import get_emails

	email_records = get_emails(
		count=count,
		start_date=start_date,
		end_date=end_date,
		service=service,
	)
	store_emails(db, gmail_account_id, email_records, service=service)
	return email_records


def get_or_create_gmail_account(db: Session, service) -> GmailAccount:
	"""Find the authenticated Gmail account or create its database records."""
	from app.services.get_mail import TOKEN_FILE

	profile = service.users().getProfile(userId="me").execute()
	email_address = profile["emailAddress"]
	account = db.scalar(
		select(GmailAccount).where(GmailAccount.email == email_address)
	)
	if account is not None:
		return account

	user = db.scalar(select(User).where(User.email == email_address))
	if user is None:
		user = User(google_id=email_address, email=email_address)
		db.add(user)
		db.flush()

	token_data = {}
	try:
		with open(TOKEN_FILE, encoding="utf-8") as token_file:
			token_data = __import__("json").load(token_file)
	except (OSError, ValueError):
		pass

	account = GmailAccount(
		user_id=user.id,
		email=email_address,
		access_token=token_data.get("token", "oauth-managed"),
		refresh_token=token_data.get("refresh_token"),
		token_expiry=None,
		scope=" ".join(token_data.get("scopes", [])),
	)
	db.add(account)
	db.commit()
	db.refresh(account)
	return account

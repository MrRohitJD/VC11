"""Gmail sync service — builds authenticated Gmail API clients from stored OAuth
tokens and provides full-mailbox sync with progress reporting."""

import json
import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from uuid import UUID

from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.email import Email, EmailLabel, GmailAccount, Label, User
from app.services.get_mail import clean_subject, extract_email_full, get_email_body

logger = logging.getLogger(__name__)

GMAIL_SCOPES = [
	"openid",
	"https://www.googleapis.com/auth/userinfo.email",
	"https://www.googleapis.com/auth/userinfo.profile",
	"https://www.googleapis.com/auth/gmail.readonly",
]


# ------------------------------------------------------------------
# Build authenticated Gmail service from stored OAuth tokens
# ------------------------------------------------------------------

def build_gmail_service_for_user(db: Session, user: User):
	"""Create a Gmail API service using the user's stored OAuth credentials.

	Automatically refreshes expired access tokens and persists new tokens
	back to the database.
	"""
	account = db.scalar(
		select(GmailAccount).where(GmailAccount.user_id == user.id)
	)
	if account is None:
		raise ValueError("No Gmail account linked to this user")

	creds = Credentials(
		token=account.access_token,
		refresh_token=account.refresh_token,
		token_uri="https://oauth2.googleapis.com/token",
		client_id=settings.google_client_id,
		client_secret=settings.google_client_secret,
		scopes=account.scope.split(" ") if account.scope else GMAIL_SCOPES,
	)

	if creds.expired and creds.refresh_token:
		creds.refresh(GoogleAuthRequest())
		account.access_token = creds.token
		if creds.refresh_token:
			account.refresh_token = creds.refresh_token
		account.token_expiry = creds.expiry
		account.updated_at = datetime.now(timezone.utc)
		db.commit()

	return build("gmail", "v1", credentials=creds)


# ------------------------------------------------------------------
# Rate-limited Gmail API helper
# ------------------------------------------------------------------

_MAX_RETRIES = 5


def _call_with_retry(request_fn, *, retries=_MAX_RETRIES):
	"""Execute a Gmail API request with exponential backoff on transient errors."""
	for attempt in range(retries):
		try:
			return request_fn()
		except HttpError as exc:
			if exc.resp.status in (429, 500, 503) and attempt < retries - 1:
				wait = min(2 ** attempt + 0.5, 30)
				logger.warning(
					"Gmail API %s (attempt %d/%d), retrying in %.1fs",
					exc.resp.status, attempt + 1, retries, wait,
				)
				time.sleep(wait)
			else:
				raise


# ------------------------------------------------------------------
# Sync all Gmail labels
# ------------------------------------------------------------------

def sync_all_labels(
	db: Session,
	gmail_account_id: UUID,
	service,
) -> dict[str, Label]:
	"""Upsert every Gmail system and user label for an account."""
	label_data_by_id: dict[str, dict] = {}
	page_token = None
	while True:
		request_args: dict = {"userId": "me"}
		if page_token:
			request_args["pageToken"] = page_token
		response = _call_with_retry(
			lambda args=request_args: service.users().labels().list(**args).execute()
		)
		for label_data in response.get("labels", []):
			label_id = label_data.get("id")
			if label_id:
				label_data_by_id[label_id] = label_data
		page_token = response.get("nextPageToken")
		if not page_token:
			break

	labels: dict[str, Label] = {}
	for label_id, label_data in label_data_by_id.items():
		label = db.scalar(
			select(Label).where(
				Label.gmail_account_id == gmail_account_id,
				Label.label_id == label_id,
			)
		)
		if label is None:
			label = Label(
				gmail_account_id=gmail_account_id,
				label_id=label_id,
				name=label_data.get("name", label_id),
				type=label_data.get("type"),
			)
			db.add(label)
		else:
			label.name = label_data.get("name", label_id)
			label.type = label_data.get("type")
		db.flush()
		labels[label_id] = label

	db.commit()
	return labels


# ------------------------------------------------------------------
# List all message IDs (paginated)
# ------------------------------------------------------------------

def _list_all_message_ids(service, *, progress_callback=None) -> list[str]:
	"""Fetch every message ID in the user's mailbox using pagination."""
	message_ids: list[str] = []
	page_token = None

	while True:
		kwargs: dict = {"userId": "me", "maxResults": 500}
		if page_token:
			kwargs["pageToken"] = page_token

		response = _call_with_retry(
			lambda kw=kwargs: service.users().messages().list(**kw).execute()
		)
		for msg in response.get("messages", []):
			message_ids.append(msg["id"])

		if progress_callback:
			progress_callback("listing", len(message_ids), 0)

		page_token = response.get("nextPageToken")
		if not page_token:
			break

	return message_ids


# ------------------------------------------------------------------
# Batch-fetch full message details
# ------------------------------------------------------------------

def _batch_fetch_messages(
	service,
	message_ids: list[str],
	*,
	batch_size: int = 50,
	progress_callback=None,
) -> list[dict]:
	"""Fetch full message details in batches using the Gmail batch API."""
	all_messages: list[dict] = []

	for batch_start in range(0, len(message_ids), batch_size):
		batch_ids = message_ids[batch_start: batch_start + batch_size]
		batch_results: list[dict] = []

		def make_callback(results_list):
			def callback(request_id, response, exception):
				if exception:
					logger.warning("Batch fetch error for %s: %s", request_id, exception)
				elif response:
					results_list.append(response)
			return callback

		batch = service.new_batch_http_request(callback=make_callback(batch_results))
		for msg_id in batch_ids:
			batch.add(
				service.users().messages().get(userId="me", id=msg_id, format="full"),
				request_id=msg_id,
			)

		_call_with_retry(batch.execute)
		all_messages.extend(batch_results)

		if progress_callback:
			progress_callback("fetching", len(message_ids), len(all_messages))

	return all_messages


# ------------------------------------------------------------------
# Parse a raw Gmail API message into a storable dict
# ------------------------------------------------------------------

def _parse_message(message: dict) -> dict:
	"""Extract all fields from a Gmail API message response."""
	payload = message.get("payload", {})
	headers = {
		h.get("name", "").lower(): h.get("value", "")
		for h in payload.get("headers", [])
	}

	# Attachment metadata
	attachments = []
	def _scan_attachments(part):
		body = part.get("body", {})
		if body.get("attachmentId"):
			attachments.append({
				"filename": part.get("filename", ""),
				"mimeType": part.get("mimeType", ""),
				"size": body.get("size", 0),
				"attachmentId": body.get("attachmentId"),
			})
		for sub in part.get("parts", []):
			_scan_attachments(sub)

	_scan_attachments(payload)

	label_ids = message.get("labelIds", [])

	return {
		"id": message.get("id", ""),
		"threadId": message.get("threadId", ""),
		"labelIds": label_ids,
		"from": headers.get("from", ""),
		"to": headers.get("to", ""),
		"cc": headers.get("cc", ""),
		"bcc": headers.get("bcc", ""),
		"subject": clean_subject(headers.get("subject", "")),
		"body": get_email_body(payload),
		"snippet": message.get("snippet", ""),
		"date": headers.get("date", ""),
		"sizeEstimate": message.get("sizeEstimate"),
		"historyId": message.get("historyId"),
		"attachments": attachments,
		"is_read": "UNREAD" not in label_ids,
		"has_attachment": len(attachments) > 0,
	}


# ------------------------------------------------------------------
# Store parsed messages into the database
# ------------------------------------------------------------------

def _store_messages(
	db: Session,
	gmail_account_id: UUID,
	parsed_messages: list[dict],
	label_cache: dict[str, Label],
	*,
	progress_callback=None,
) -> int:
	"""Upsert parsed email records and link labels. Returns count stored."""
	stored = 0
	total = len(parsed_messages)

	for idx, record in enumerate(parsed_messages):
		gmail_msg_id = record["id"]
		if not gmail_msg_id:
			continue

		# Skip duplicates
		existing = db.scalar(
			select(Email).where(
				Email.gmail_account_id == gmail_account_id,
				Email.gmail_message_id == gmail_msg_id,
			)
		)
		if existing is not None:
			stored += 1
			if progress_callback and (stored % 100 == 0 or stored == total):
				progress_callback("storing", total, stored)
			continue

		email = Email(
			gmail_account_id=gmail_account_id,
			gmail_message_id=gmail_msg_id,
			thread_id=record.get("threadId"),
			subject=record.get("subject"),
			body=record.get("body"),
			snippet=record.get("snippet"),
			from_email=record.get("from"),
			to_email=record.get("to"),
			cc=record.get("cc"),
			bcc=record.get("bcc"),
			received_at=_parse_received_at(record.get("date")),
			is_read=record.get("is_read", False),
			has_attachment=record.get("has_attachment", False),
			attachments_meta=json.dumps(record.get("attachments", [])) if record.get("attachments") else None,
			size_estimate=record.get("sizeEstimate"),
			history_id=record.get("historyId"),
		)
		db.add(email)
		db.flush()

		# Link labels
		for gmail_label_id in dict.fromkeys(record.get("labelIds", [])):
			label = label_cache.get(gmail_label_id)
			if label is None:
				label = db.scalar(
					select(Label).where(
						Label.gmail_account_id == gmail_account_id,
						Label.label_id == gmail_label_id,
					)
				)
			if label is not None:
				db.add(EmailLabel(email_id=email.id, label_id=label.id))

		stored += 1
		# Commit in batches of 200 to avoid huge transactions
		if stored % 200 == 0:
			db.commit()
			if progress_callback:
				progress_callback("storing", total, stored)

	db.commit()
	if progress_callback:
		progress_callback("storing", total, stored)
	return stored


# ------------------------------------------------------------------
# Full sync orchestrator
# ------------------------------------------------------------------

def sync_all_emails(
	db: Session,
	user: User,
	service,
	*,
	progress_callback=None,
) -> dict:
	"""Full mailbox sync: list → batch fetch → parse → store.

	progress_callback(stage, total, done) is called throughout:
	  stage = "listing" | "fetching" | "storing" | "labels" | "completed"
	"""
	account = db.scalar(
		select(GmailAccount).where(GmailAccount.user_id == user.id)
	)
	if account is None:
		raise ValueError("No Gmail account linked to this user")

	# 1. Sync labels first
	if progress_callback:
		progress_callback("labels", 0, 0)
	label_cache = sync_all_labels(db, account.id, service)
	labels_count = len(label_cache)
	if progress_callback:
		progress_callback("labels", labels_count, labels_count)

	# 2. List all message IDs
	if progress_callback:
		progress_callback("listing", 0, 0)
	message_ids = _list_all_message_ids(service, progress_callback=progress_callback)
	total_found = len(message_ids)

	if total_found == 0:
		if progress_callback:
			progress_callback("completed", 0, 0)
		return {"emails_found": 0, "emails_synced": 0, "labels_synced": labels_count}

	# 3. Filter out already-stored message IDs for efficiency
	existing_ids = set(
		db.scalars(
			select(Email.gmail_message_id).where(
				Email.gmail_account_id == account.id,
				Email.gmail_message_id.in_(message_ids),
			)
		).all()
	)
	new_ids = [mid for mid in message_ids if mid not in existing_ids]
	logger.info("Found %d messages, %d new to fetch", total_found, len(new_ids))

	# 4. Batch-fetch full details for new messages only
	if new_ids:
		if progress_callback:
			progress_callback("fetching", len(new_ids), 0)
		raw_messages = _batch_fetch_messages(
			service, new_ids, progress_callback=progress_callback,
		)

		# 5. Parse
		parsed = [_parse_message(msg) for msg in raw_messages]

		# 6. Store
		if progress_callback:
			progress_callback("storing", len(parsed), 0)
		emails_stored = _store_messages(
			db, account.id, parsed, label_cache, progress_callback=progress_callback,
		)
	else:
		emails_stored = 0

	if progress_callback:
		progress_callback("completed", total_found, emails_stored)

	return {
		"emails_found": total_found,
		"emails_synced": emails_stored + len(existing_ids),
		"labels_synced": labels_count,
	}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_received_at(value: str | None) -> datetime | None:
	if not value:
		return None
	try:
		return parsedate_to_datetime(value)
	except (TypeError, ValueError):
		return None


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
			token_data = json.load(token_file)
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


# ------------------------------------------------------------------
# Legacy compat: fetch_and_store_emails (used by old main.py)
# ------------------------------------------------------------------

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
	stored = _store_emails_legacy(db, gmail_account_id, email_records, service=service)
	return email_records


def _store_emails_legacy(
	db: Session,
	gmail_account_id: UUID,
	email_records: list[dict],
	service=None,
) -> list[Email]:
	"""Legacy insert for the old get_emails format."""
	label_cache = sync_all_labels(db, gmail_account_id, service) if service else {}
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

		# Link labels
		from sqlalchemy import delete as sa_delete
		sa_delete_stmt = sa_delete(EmailLabel).where(EmailLabel.email_id == email.id)
		db.execute(sa_delete_stmt)
		for label_id in dict.fromkeys(record.get("labelIds", [])):
			label = label_cache.get(label_id)
			if label is None:
				label = db.scalar(
					select(Label).where(
						Label.gmail_account_id == gmail_account_id,
						Label.label_id == label_id,
					)
				)
			if label is None and service:
				try:
					label_data = service.users().labels().get(
						userId="me", id=label_id,
					).execute()
					label = Label(
						gmail_account_id=gmail_account_id,
						label_id=label_id,
						name=label_data.get("name", label_id),
						type=label_data.get("type"),
					)
					db.add(label)
					db.flush()
				except Exception:
					continue
			if label is not None:
				db.add(EmailLabel(email_id=email.id, label_id=label.id))

		stored.append(email)

	db.commit()
	for email in stored:
		db.refresh(email)
	return stored

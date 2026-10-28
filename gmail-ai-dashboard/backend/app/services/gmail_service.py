"""Gmail sync service — authenticate users and synchronize their Inbox."""

import json
import logging
import time
import base64
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import getaddresses, parsedate_to_datetime
import re
from uuid import UUID

from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.connection import SessionLocal
from app.models.email import Email, EmailLabel, GmailAccount, Label, User
from app.services.get_mail import clean_subject, extract_email_full, get_email_body, get_email_body_html

logger = logging.getLogger(__name__)

GMAIL_SCOPES = [
	"openid",
	"https://www.googleapis.com/auth/userinfo.email",
	"https://www.googleapis.com/auth/userinfo.profile",
	"https://www.googleapis.com/auth/gmail.readonly",
	"https://www.googleapis.com/auth/gmail.send",
]
GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"


# ------------------------------------------------------------------
# Build authenticated Gmail service from stored OAuth tokens
# ------------------------------------------------------------------

def build_gmail_service_for_user(db: Session, user: User, *, timeout_seconds: float | None = None):
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
	# Without expiry, google-auth treats a stored expired token as valid and
	# the first live Gmail search fails instead of refreshing the OAuth session.
	if account.token_expiry:
		creds.expiry = account.token_expiry.astimezone(timezone.utc).replace(tzinfo=None) if account.token_expiry.tzinfo else account.token_expiry

	if creds.expired and creds.refresh_token:
		refresh_request = GoogleAuthRequest()
		if timeout_seconds is not None:
			from functools import partial
			refresh_request = partial(refresh_request, timeout=timeout_seconds)
		creds.refresh(refresh_request)
		account.access_token = creds.token
		if creds.refresh_token:
			account.refresh_token = creds.refresh_token
		account.token_expiry = creds.expiry
		account.updated_at = datetime.now(timezone.utc)
		db.commit()

	if timeout_seconds is not None:
		import httplib2
		from google_auth_httplib2 import AuthorizedHttp
		http = AuthorizedHttp(creds, http=httplib2.Http(timeout=timeout_seconds))
		return build("gmail", "v1", http=http, static_discovery=True, cache_discovery=False)
	return build("gmail", "v1", credentials=creds)


def send_gmail_message(
	db: Session,
	user: User,
	recipient: str,
	subject: str | None,
	body: str,
	cc: str | None = None,
	bcc: str | None = None,
) -> dict:
	"""Deliver one approved message through the user's authenticated Gmail API."""
	account = db.scalar(select(GmailAccount).where(GmailAccount.user_id == user.id))
	if account is None:
		raise ValueError("No Gmail account linked to this user")
	scopes = set((account.scope or "").split())
	if scopes and GMAIL_SEND_SCOPE not in scopes:
		raise PermissionError("Gmail send permission is missing; reconnect Gmail to enable sending")
	to_header = normalize_recipient_header(recipient, "recipient", required=True)
	cc_header = normalize_recipient_header(cc, "Cc")
	bcc_header = normalize_recipient_header(bcc, "Bcc")

	message = EmailMessage()
	message["To"] = to_header
	if cc_header:
		message["Cc"] = cc_header
	if bcc_header:
		message["Bcc"] = bcc_header
	message["Subject"] = (subject or "").strip()
	message.set_content(body or "")
	raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
	service = build_gmail_service_for_user(db, user)
	return _call_with_retry(
		lambda: service.users().messages().send(
			userId="me", body={"raw": raw}
		).execute()
	)


def normalize_recipient_header(value: str | None, field_name: str, *, required: bool = False) -> str:
	"""Validate and normalize one or more comma/semicolon-separated addresses."""
	raw = str(value or "").strip()
	if not raw:
		if required:
			raise ValueError("A valid recipient email address is required")
		return ""
	if "\r" in raw or "\n" in raw:
		raise ValueError(f"{field_name} contains an invalid line break")

	entries = getaddresses([raw.replace(";", ",")])
	addresses = [address.strip() for _, address in entries]
	if not addresses or any(
		not re.fullmatch(r"[^@\s,<>]+@[^@\s,<>]+\.[^@\s,<>]+", address)
		for address in addresses
	):
		raise ValueError(f"Enter valid email address(es) for {field_name}")
	return ", ".join(addresses)


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
			if exc.resp.status in (403, 429, 500, 503) and attempt < retries - 1:
				wait = min(5 * (2 ** attempt), 60)
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
# List one bounded page of Inbox message IDs
# ------------------------------------------------------------------

def _list_inbox_page(service, *, page_token: str | None = None, page_size: int = 500) -> tuple[list[str], str | None]:
	"""Return one Gmail Inbox page without accumulating the whole mailbox in RAM."""
	kwargs: dict = {"userId": "me", "labelIds": ["INBOX"], "maxResults": page_size}
	if page_token:
		kwargs["pageToken"] = page_token
	response = _call_with_retry(
		lambda: service.users().messages().list(**kwargs).execute()
	)
	return [message["id"] for message in response.get("messages", [])], response.get("nextPageToken")


# ------------------------------------------------------------------
# Batch-fetch full message details
# ------------------------------------------------------------------

def _batch_fetch_messages(
	service,
	message_ids: list[str],
	*,
	batch_size: int = 20,
	progress_callback=None,
) -> list[dict]:
	"""Fetch full message details in batches using the Gmail batch API."""
	all_messages: list[dict] = []

	for batch_start in range(0, len(message_ids), batch_size):
		batch_ids = message_ids[batch_start: batch_start + batch_size]
		batch_results: list[dict] = []

		def execute_batch():
			batch_results.clear()
			batch_errors: list[Exception] = []

			def callback(request_id, response, exception):
				if exception:
					batch_errors.append(exception)
				elif response:
					batch_results.append(response)

			batch = service.new_batch_http_request(callback=callback)
			for msg_id in batch_ids:
				batch.add(
					service.users().messages().get(userId="me", id=msg_id, format="full"),
					request_id=msg_id,
				)
			batch.execute()
			if batch_errors:
				transient_error = next(
					(error for error in batch_errors if isinstance(error, HttpError) and error.resp.status in (403, 429, 500, 503)),
					None,
				)
				if transient_error is not None:
					raise transient_error
				for error in batch_errors:
					# A message may be deleted between list and get. Other failures
					# must stop the scan so the next run can retry the whole page.
					if isinstance(error, HttpError) and error.resp.status == 404:
						logger.info("Inbox message disappeared during sync: %s", error)
						continue
					raise error

		_call_with_retry(execute_batch)
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
		"body_html": get_email_body_html(payload),
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
			if record.get("body_html") and not existing.body_html:
				existing.body_html = record["body_html"]
			if record.get("body") and not existing.body:
				existing.body = record["body"]
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
			body_html=record.get("body_html"),
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
				exists_link = db.get(EmailLabel, (email.id, label.id))
				if exists_link is None:
					db.add(EmailLabel(email_id=email.id, label_id=label.id, gmail_account_id=gmail_account_id))

		stored += 1
		# Commit in batches of 100 to avoid huge transactions
		if stored % 100 == 0:
			db.commit()
			if progress_callback:
				progress_callback("storing", total, stored)

	db.commit()
	if progress_callback:
		progress_callback("storing", total, stored)
	return stored


# ------------------------------------------------------------------
# Bounded Inbox batches and background pagination
# ------------------------------------------------------------------

def _sync_inbox_batch(
	db: Session,
	account_id: UUID,
	service,
	label_cache: dict[str, Label],
	message_ids: list[str],
	*,
	progress_callback=None,
) -> tuple[int, int]:
	"""Store missing Inbox messages and restore the Inbox link on older records."""
	if not message_ids:
		return 0, 0
	rows = db.execute(
		select(Email.id, Email.gmail_message_id).where(
			Email.gmail_account_id == account_id,
			Email.gmail_message_id.in_(message_ids),
		)
	).all()
	existing_by_message_id = {message_id: email_id for email_id, message_id in rows}
	inbox_label = label_cache.get("INBOX")
	if inbox_label and existing_by_message_id:
		existing_email_ids = list(existing_by_message_id.values())
		linked_ids = set(db.scalars(
			select(EmailLabel.email_id).where(
				EmailLabel.gmail_account_id == account_id,
				EmailLabel.label_id == inbox_label.id,
				EmailLabel.email_id.in_(existing_email_ids),
			)
		).all())
		for email_id in existing_email_ids:
			if email_id not in linked_ids:
				db.add(EmailLabel(email_id=email_id, label_id=inbox_label.id, gmail_account_id=account_id))
		db.commit()

	new_ids = [message_id for message_id in message_ids if message_id not in existing_by_message_id]
	if not new_ids:
		return 0, len(existing_by_message_id)
	raw_messages = _batch_fetch_messages(service, new_ids, progress_callback=progress_callback)
	parsed = [_parse_message(message) for message in raw_messages]
	stored = _store_messages(db, account_id, parsed, label_cache, progress_callback=progress_callback)
	return stored, len(existing_by_message_id)


def _schedule_ai_processing(account_id: UUID) -> None:
	"""Queue classification and indexing after a committed Inbox batch."""
	try:
		from app.services.llm_classification_service import enqueue_classification_job, trigger_immediate_classification
		enqueue_classification_job(account_id, job_type="INCREMENTAL")
		trigger_immediate_classification()
		from app.services.rag_auto_indexer import trigger_immediate_auto_indexing
		trigger_immediate_auto_indexing()
	except Exception as exc:
		logger.warning("Background AI processing trigger failed: %s", exc)


def _sync_remaining_emails_worker(
	user_id: UUID,
	account_id: UUID,
	page_token: str,
	*,
	progress_callback=None,
	batch_chunk_size: int = 100,
):
	"""Walk every remaining Inbox page while keeping memory use bounded."""
	seen = 0
	stored = 0
	cursor_reset = False
	try:
		while page_token:
			worker_db = SessionLocal()
			try:
				worker_user = worker_db.get(User, user_id)
				if worker_user is None:
					raise ValueError("User was removed during Inbox sync")
				service = build_gmail_service_for_user(worker_db, worker_user, timeout_seconds=60)
				try:
					message_ids, next_page_token = _list_inbox_page(service, page_token=page_token)
				except HttpError as exc:
					if exc.resp.status != 400 or cursor_reset:
						raise
					# Gmail can invalidate a saved cursor after a process restart.
					# Restart from page two; existing messages are skipped in batches.
					cursor_reset = True
					account = worker_db.get(GmailAccount, account_id)
					if account:
						account.inbox_sync_page_token = None
						worker_db.commit()
					_, page_token = _list_inbox_page(service, page_size=100)
					continue
				label_cache = {
					label.label_id: label
					for label in worker_db.scalars(
						select(Label).where(Label.gmail_account_id == account_id)
					).all()
				}
				page_new_count = 0
				for start in range(0, len(message_ids), batch_chunk_size):
					chunk_ids = message_ids[start:start + batch_chunk_size]
					new_count, _ = _sync_inbox_batch(worker_db, account_id, service, label_cache, chunk_ids)
					page_new_count += new_count
					stored += new_count
					seen += len(chunk_ids)
					account = worker_db.get(GmailAccount, account_id)
					if account:
						account.inbox_sync_heartbeat_at = datetime.now(timezone.utc)
						worker_db.commit()
					if progress_callback:
						progress_callback("syncing", seen, stored)
				if page_new_count:
					_schedule_ai_processing(account_id)
				account = worker_db.get(GmailAccount, account_id)
				if account:
					account.inbox_sync_page_token = next_page_token
					account.inbox_sync_heartbeat_at = datetime.now(timezone.utc)
					worker_db.commit()
				page_token = next_page_token
			finally:
				worker_db.close()

		with SessionLocal() as worker_db:
			account = worker_db.get(GmailAccount, account_id)
			if account:
				account.sync_status = "IDLE"
				account.inbox_sync_page_token = None
				account.inbox_sync_heartbeat_at = None
				account.last_synced_at = datetime.now(timezone.utc)
				worker_db.commit()
		if progress_callback:
			progress_callback("completed", seen, stored)
		logger.info("Finished Inbox backfill for account %s: %d scanned, %d stored", account_id, seen, stored)
	except Exception as exc:
		logger.exception("Inbox background sync failed for account %s", account_id)
		try:
			with SessionLocal() as worker_db:
				account = worker_db.get(GmailAccount, account_id)
				if account:
					account.sync_status = "BACKFILL_ERROR"
					account.inbox_sync_heartbeat_at = None
					worker_db.commit()
		except Exception:
			logger.exception("Unable to save Inbox sync failure for account %s", account_id)
		if progress_callback:
			progress_callback("error", seen, stored, str(exc))


# ------------------------------------------------------------------
# Initial Inbox sync orchestrator
# ------------------------------------------------------------------

def sync_all_emails(
	db: Session,
	user: User,
	service,
	*,
	initial_limit: int = 100,
	progress_callback=None,
	background_callback=None,
) -> dict:
	"""Store the first 100 Inbox messages, then page through the rest in background."""
	import threading

	account = db.scalar(
		select(GmailAccount).where(GmailAccount.user_id == user.id)
	)
	if account is None:
		raise ValueError("No Gmail account linked to this user")
	account.sync_status = "INITIAL"
	account.inbox_sync_heartbeat_at = datetime.now(timezone.utc)
	db.commit()

	# 1. Sync labels first
	if progress_callback:
		progress_callback("labels", 0, 0)
	label_cache = sync_all_labels(db, account.id, service)
	labels_count = len(label_cache)
	if progress_callback:
		progress_callback("labels", labels_count, labels_count)

	# 2. List only the first Inbox page before the dashboard opens.
	if progress_callback:
		progress_callback("listing", 0, 0)
	message_ids, next_page_token = _list_inbox_page(service, page_size=initial_limit)
	resume_token = account.inbox_sync_page_token or next_page_token
	if progress_callback:
		progress_callback("listing", len(message_ids), 0)
		progress_callback("fetching", len(message_ids), 0)
	emails_stored, already_stored = _sync_inbox_batch(
		db, account.id, service, label_cache, message_ids,
		progress_callback=progress_callback,
	)
	initial_synced = emails_stored + already_stored
	if resume_token:
		account.sync_status = "BACKFILL"
		account.inbox_sync_page_token = resume_token
		account.inbox_sync_heartbeat_at = datetime.now(timezone.utc)
		db.commit()
		if background_callback:
			background_callback("syncing", 0, 0)
		bg_thread = threading.Thread(
			target=_sync_remaining_emails_worker,
			args=(user.id, account.id, resume_token),
			kwargs={"progress_callback": background_callback},
			name=f"inbox-sync-{str(account.id)[:8]}",
			daemon=True,
		)
		bg_thread.start()
	else:
		account.last_synced_at = datetime.now(timezone.utc)
		account.sync_status = "IDLE"
		account.inbox_sync_page_token = None
		account.inbox_sync_heartbeat_at = None
		db.commit()
		if background_callback:
			background_callback("completed", 0, 0)

	if progress_callback:
		progress_callback("completed", len(message_ids), initial_synced)

	return {
		"emails_found": len(message_ids),
		"emails_synced": initial_synced,
		"labels_synced": labels_count,
		"background_syncing": bool(resume_token),
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
				exists_link = db.get(EmailLabel, (email.id, label.id))
				if exists_link is None:
					db.add(EmailLabel(email_id=email.id, label_id=label.id, gmail_account_id=gmail_account_id))

		stored.append(email)

	db.commit()
	for email in stored:
		db.refresh(email)
	return stored

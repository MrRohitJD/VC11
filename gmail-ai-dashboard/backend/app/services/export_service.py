"""Email export service — export emails from the database as CSV or JSON.

Provides high-performance data export by reading directly from the local
database instead of hitting the Gmail API.  Users who have already synced
their emails can export/download their mail data instantly.
"""

import csv
import io
import json
import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.models.email import Email, EmailClassification, EmailLabel, GmailAccount, Label

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Serialise a single Email row to a flat dict for export
# ------------------------------------------------------------------

def _email_to_export_dict(
	email: Email,
	*,
	include_body: bool = False,
	include_html: bool = False,
) -> dict:
	"""Convert an Email ORM object into a JSON-serializable dictionary."""
	record: dict = {
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
			el.label.name for el in email.labels if el.label
		],
	}
	if include_body:
		record["body"] = email.body
	if include_html:
		record["body_html"] = email.body_html
	if email.attachments_meta:
		try:
			record["attachments"] = json.loads(email.attachments_meta)
		except (json.JSONDecodeError, TypeError):
			record["attachments"] = []
	else:
		record["attachments"] = []
	return record


# ------------------------------------------------------------------
# Export as JSON (with metadata envelope)
# ------------------------------------------------------------------

def export_emails_json(
	db: Session,
	gmail_account_id: UUID,
	*,
	include_body: bool = False,
	include_html: bool = False,
	limit: int | None = None,
) -> dict:
	"""Build a complete JSON export with metadata and email records.

	Returns a dict suitable for ``json.dumps()`` or direct API response.
	"""
	account = db.get(GmailAccount, gmail_account_id)
	if account is None:
		raise ValueError("Gmail account not found")

	query = (
		select(Email)
		.where(Email.gmail_account_id == gmail_account_id)
		.order_by(Email.received_at.desc().nullslast())
	)
	if limit:
		query = query.limit(limit)

	emails = db.scalars(query).all()
	total_in_db = db.scalar(
		select(func.count(Email.id)).where(Email.gmail_account_id == gmail_account_id)
	) or 0

	records = [
		_email_to_export_dict(e, include_body=include_body, include_html=include_html)
		for e in emails
	]

	return {
		"metadata": {
			"account_email": account.email,
			"exported_at": datetime.now(timezone.utc).isoformat(),
			"total_emails_in_database": total_in_db,
			"emails_in_export": len(records),
			"includes_body": include_body,
			"includes_html": include_html,
		},
		"emails": records,
	}


# ------------------------------------------------------------------
# Export as CSV (streaming generator)
# ------------------------------------------------------------------

_CSV_COLUMNS = [
	"gmail_message_id",
	"thread_id",
	"subject",
	"snippet",
	"from_email",
	"to_email",
	"cc",
	"bcc",
	"received_at",
	"is_read",
	"has_attachment",
	"size_estimate",
	"labels",
]


def export_emails_csv(
	db: Session,
	gmail_account_id: UUID,
	*,
	include_body: bool = False,
	limit: int | None = None,
) -> str:
	"""Build a CSV string of email records from the database.

	Returns the complete CSV content as a string.
	"""
	columns = list(_CSV_COLUMNS)
	if include_body:
		columns.append("body")

	query = (
		select(Email)
		.where(Email.gmail_account_id == gmail_account_id)
		.order_by(Email.received_at.desc().nullslast())
	)
	if limit:
		query = query.limit(limit)

	emails = db.scalars(query).all()

	output = io.StringIO()
	writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
	writer.writeheader()

	for email in emails:
		row = _email_to_export_dict(email, include_body=include_body)
		# Flatten labels list to semicolon-separated string for CSV
		row["labels"] = "; ".join(row.get("labels", []))
		writer.writerow(row)

	return output.getvalue()


# ------------------------------------------------------------------
# Database stats — quick summary of what's available for export
# ------------------------------------------------------------------

def get_export_stats(db: Session, gmail_account_id: UUID) -> dict:
	"""Return counts and date range for the user's stored email data."""
	base = Email.gmail_account_id == gmail_account_id

	total = db.scalar(select(func.count(Email.id)).where(base)) or 0
	with_body = db.scalar(
		select(func.count(Email.id)).where(base, Email.body.is_not(None))
	) or 0
	with_html = db.scalar(
		select(func.count(Email.id)).where(base, Email.body_html.is_not(None))
	) or 0
	oldest = db.scalar(select(func.min(Email.received_at)).where(base))
	newest = db.scalar(select(func.max(Email.received_at)).where(base))
	labels_count = db.scalar(
		select(func.count(Label.id)).where(Label.gmail_account_id == gmail_account_id)
	) or 0

	return {
		"total_emails": total,
		"emails_with_body": with_body,
		"emails_with_html_body": with_html,
		"labels_count": labels_count,
		"oldest_email": oldest.isoformat() if oldest else None,
		"newest_email": newest.isoformat() if newest else None,
	}


# ------------------------------------------------------------------
# Import emails from JSON or CSV into the database
# ------------------------------------------------------------------

def import_emails_from_json(
	db: Session,
	gmail_account_id: UUID,
	json_content: str,
) -> dict:
	"""Parse an exported JSON file and store emails into the database.

	Handles both envelope format ({"metadata": ..., "emails": [...]})
	and bare list format ([{...}]).
	"""
	data = json.loads(json_content)
	if isinstance(data, dict) and "emails" in data:
		email_records = data["emails"]
	elif isinstance(data, list):
		email_records = data
	else:
		raise ValueError("Invalid JSON format: expected list of emails or object with 'emails' array")

	return _store_imported_email_records(db, gmail_account_id, email_records)


def import_emails_from_csv(
	db: Session,
	gmail_account_id: UUID,
	csv_content: str,
) -> dict:
	"""Parse an exported CSV file and store emails into the database."""
	f = io.StringIO(csv_content)
	reader = csv.DictReader(f)
	email_records = list(reader)
	return _store_imported_email_records(db, gmail_account_id, email_records)


def _store_imported_email_records(
	db: Session,
	gmail_account_id: UUID,
	records: list[dict],
) -> dict:
	"""Store parsed email records into the database, linking labels."""
	imported = 0
	skipped = 0

	# Preload existing labels for this account
	existing_labels = db.scalars(
		select(Label).where(Label.gmail_account_id == gmail_account_id)
	).all()
	label_by_name = {l.name: l for l in existing_labels}

	for r in records:
		msg_id = r.get("gmail_message_id") or r.get("id")
		if not msg_id:
			skipped += 1
			continue

		# Check duplicate
		existing = db.scalar(
			select(Email).where(
				Email.gmail_account_id == gmail_account_id,
				Email.gmail_message_id == msg_id,
			)
		)

		if existing:
			# Update body if missing
			if r.get("body") and not existing.body:
				existing.body = r["body"]
			if r.get("body_html") and not existing.body_html:
				existing.body_html = r["body_html"]
			skipped += 1
			continue

		# Parse received_at datetime
		received_at = None
		raw_date = r.get("received_at") or r.get("date")
		if raw_date:
			try:
				if isinstance(raw_date, str):
					received_at = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
			except Exception:
				pass

		is_read = r.get("is_read")
		if isinstance(is_read, str):
			is_read = is_read.lower() in ("true", "1", "yes")
		elif not isinstance(is_read, bool):
			is_read = False

		has_attachment = r.get("has_attachment")
		if isinstance(has_attachment, str):
			has_attachment = has_attachment.lower() in ("true", "1", "yes")
		elif not isinstance(has_attachment, bool):
			has_attachment = False

		size_est = r.get("size_estimate")
		try:
			size_est = int(size_est) if size_est is not None and str(size_est).isdigit() else None
		except Exception:
			size_est = None

		email = Email(
			gmail_account_id=gmail_account_id,
			gmail_message_id=msg_id,
			thread_id=r.get("thread_id") or r.get("threadId"),
			subject=r.get("subject"),
			body=r.get("body"),
			body_html=r.get("body_html"),
			snippet=r.get("snippet"),
			from_email=r.get("from_email") or r.get("from"),
			to_email=r.get("to_email") or r.get("to"),
			cc=r.get("cc"),
			bcc=r.get("bcc"),
			received_at=received_at,
			is_read=is_read,
			has_attachment=has_attachment,
			attachments_meta=json.dumps(r.get("attachments", [])) if r.get("attachments") else None,
			size_estimate=size_est,
			history_id=r.get("history_id") or r.get("historyId"),
		)
		db.add(email)
		db.flush()

		# Link labels
		raw_labels = r.get("labels", [])
		if isinstance(raw_labels, str):
			raw_labels = [lbl.strip() for lbl in raw_labels.split(";") if lbl.strip()]

		for label_name in raw_labels:
			if not label_name:
				continue
			label = label_by_name.get(label_name)
			if not label:
				label = Label(
					gmail_account_id=gmail_account_id,
					label_id=label_name.upper().replace(" ", "_"),
					name=label_name,
					type="user",
				)
				db.add(label)
				db.flush()
				label_by_name[label_name] = label

			db.add(EmailLabel(email_id=email.id, label_id=label.id, gmail_account_id=gmail_account_id))

		imported += 1
		if imported % 100 == 0:
			db.commit()

	db.commit()
	if imported > 0:
		try:
			from app.services.llm_classification_service import trigger_immediate_classification
			trigger_immediate_classification()
		except Exception as exc:
			logger.warning("Could not trigger AI classification after import: %s", exc)

	return {"imported": imported, "skipped": skipped, "total": len(records)}

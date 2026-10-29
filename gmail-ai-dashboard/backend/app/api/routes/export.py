"""Export routes — download emails from the database as CSV or JSON.

These endpoints serve data exclusively from the local database, so they
are fast and do NOT hit the Gmail API.  This is the recommended way for
users to bulk-access their email data after the initial sync.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.connection import get_db
from app.models.email import GmailAccount, User
from app.services.export_service import (
	export_emails_csv,
	export_emails_json,
	get_export_stats,
)
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/export", tags=["export"])


def _get_account(db: Session, user: User) -> GmailAccount:
	account = db.scalar(
		select(GmailAccount).where(GmailAccount.user_id == user.id)
	)
	if account is None:
		raise HTTPException(status_code=404, detail="No Gmail account linked")
	return account


@router.get("/stats")
def export_stats(
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""Return a summary of what email data is available for export."""
	account = _get_account(db, user)
	return get_export_stats(db, account.id)


@router.get("/json")
def export_json(
	include_body: bool = Query(False, description="Include plain-text body"),
	include_html: bool = Query(False, description="Include HTML body"),
	limit: int | None = Query(None, ge=1, le=50000, description="Max emails to export"),
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""Export emails as a JSON file with metadata.

	Returns a JSON document containing an envelope with export metadata
	(account, timestamp, counts) and an ``emails`` array of records.
	"""
	account = _get_account(db, user)
	data = export_emails_json(
		db, account.id,
		include_body=include_body,
		include_html=include_html,
		limit=limit,
	)
	content = json.dumps(data, indent=2, ensure_ascii=False)
	filename = f"emails_{account.email}_{data['metadata']['exported_at'][:10]}.json"

	return Response(
		content=content,
		media_type="application/json",
		headers={
			"Content-Disposition": f'attachment; filename="{filename}"',
		},
	)


@router.get("/csv")
def export_csv(
	include_body: bool = Query(False, description="Include plain-text body column"),
	limit: int | None = Query(None, ge=1, le=50000, description="Max emails to export"),
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""Export emails as a CSV file.

	Downloads a CSV with columns: gmail_message_id, thread_id, subject,
	snippet, from_email, to_email, cc, bcc, received_at, is_read,
	has_attachment, size_estimate, labels.  Optionally includes the body.
	"""
	account = _get_account(db, user)
	csv_content = export_emails_csv(
		db, account.id,
		include_body=include_body,
		limit=limit,
	)
	filename = f"emails_{account.email}.csv"

	return Response(
		content=csv_content,
		media_type="text/csv; charset=utf-8",
		headers={
			"Content-Disposition": f'attachment; filename="{filename}"',
		},
	)


@router.post("/import")
async def import_emails_file(
	request: Request,
	filename: str | None = Query(None),
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""Upload and import an email JSON or CSV file into the database.

	Accepts file data via JSON payload, raw body, or multipart form data.
	Parses the uploaded file and stores all emails directly into the local
	database without querying the Gmail API.
	"""
	from app.services.export_service import import_emails_from_csv, import_emails_from_json

	account = _get_account(db, user)
	content_type = request.headers.get("content-type", "")

	try:
		text_content = ""
		file_name = (filename or "").lower()

		if "multipart/form-data" in content_type:
			try:
				form = await request.form()
				uploaded = form.get("file")
				if uploaded and hasattr(uploaded, "read"):
					raw_bytes = await uploaded.read()
					text_content = raw_bytes.decode("utf-8-sig", errors="replace")
					file_name = (getattr(uploaded, "filename", "") or file_name).lower()
			except Exception:
				pass

		if not text_content:
			body_bytes = await request.body()
			text_content = body_bytes.decode("utf-8-sig", errors="replace")

		if not text_content.strip():
			raise HTTPException(status_code=400, detail="Uploaded file or body is empty.")

		if file_name.endswith(".json") or text_content.strip().startswith(("{", "[")):
			result = import_emails_from_json(db, account.id, text_content)
		else:
			result = import_emails_from_csv(db, account.id, text_content)

		return {
			"message": f"Successfully imported {result['imported']} emails ({result['skipped']} duplicates/skipped).",
			**result,
		}
	except Exception as exc:
		logger.error("Failed to import email file: %s", type(exc).__name__)
		raise HTTPException(status_code=400, detail="Failed to parse and store email file.") from exc

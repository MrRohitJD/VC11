"""Gmail sync routes — trigger sync, poll status, SSE stream for real-time progress."""

import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.connection import get_db, SessionLocal
from app.models.email import GmailAccount, User
from app.services.gmail_service import build_gmail_service_for_user, sync_all_emails

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/gmail", tags=["gmail"])

# In-memory sync state per user (keyed by str(user_id))
_sync_state: dict[str, dict] = {}
_SYNC_LEASE = timedelta(minutes=15)


def _get_sync_state(user_id: str, db: Session | None = None) -> dict:
	state = _sync_state.get(user_id)
	if db is not None:
		account = db.scalar(select(GmailAccount).where(GmailAccount.user_id == UUID(user_id)))
		if account is not None:
			fresh = _sync_lease_fresh(account.inbox_sync_heartbeat_at)
			if account.sync_status == "INITIAL" and fresh:
				if state and state.get("status") == "syncing":
					return state.copy()
				return {"status": "syncing", "stage": "listing", "background_status": "idle", "emails_found": 0, "emails_synced": 0, "labels_synced": 0, "error": None}
			if account.sync_status == "BACKFILL" and fresh:
				if state and state.get("background_status") == "syncing":
					return state.copy()
				return {"status": "completed", "stage": "completed", "background_status": "syncing", "background_scanned": 0, "background_synced": 0, "emails_found": 100, "emails_synced": 100, "labels_synced": 0, "error": None}
			if account.sync_status == "ERROR":
				return {"status": "error", "stage": "error", "background_status": "error", "emails_found": 0, "emails_synced": 0, "labels_synced": 0, "error": "Inbox sync failed. Please retry."}
			if account.sync_status == "BACKFILL_ERROR":
				if state and state.get("status") == "completed":
					return state.copy()
				return {"status": "completed", "stage": "completed", "background_status": "error", "background_error": "Inbox sync paused. Please retry.", "emails_found": 100, "emails_synced": 100, "labels_synced": 0, "error": None}
			if account.sync_status in ("INITIAL", "BACKFILL") and not fresh:
				state = None
	state = state or {
		"status": "idle",
		"emails_found": 0,
		"emails_synced": 0,
		"labels_synced": 0,
		"background_status": "idle",
		"background_scanned": 0,
		"background_synced": 0,
		"stage": "",
		"error": None,
	}
	return state.copy()


def _sync_lease_fresh(updated_at: datetime | None) -> bool:
	if updated_at is None:
		return False
	if updated_at.tzinfo is None:
		updated_at = updated_at.replace(tzinfo=timezone.utc)
	return updated_at >= datetime.now(timezone.utc) - _SYNC_LEASE


def _run_sync_in_thread(user_id_str: str, user_id: UUID):
	"""Run Gmail sync in a background thread (not async — Gmail API is blocking)."""
	db = SessionLocal()
	try:
		_sync_state[user_id_str].update({"status": "syncing", "stage": "authenticating"})
		user = db.get(User, user_id)
		if user is None:
			_sync_state[user_id_str] = {
				"status": "error",
				"stage": "error",
				"error": "User not found",
				"emails_found": 0,
				"emails_synced": 0,
				"labels_synced": 0,
			}
			return

		service = build_gmail_service_for_user(db, user, timeout_seconds=60)

		def progress_callback(stage, total, done):
			state = _sync_state.setdefault(user_id_str, {})
			state["status"] = "syncing"
			state["stage"] = stage
			if stage == "listing":
				state["emails_found"] = total
			elif stage == "fetching":
				state["emails_found"] = total
				state["emails_fetched"] = done
			elif stage == "storing":
				state["emails_synced"] = done
			elif stage == "labels":
				state["labels_synced"] = done
			elif stage == "completed":
				state["status"] = "completed"
				state["emails_found"] = total
				state["emails_synced"] = done

		def background_callback(status, scanned, synced, error=None):
			state = _sync_state.setdefault(user_id_str, {})
			state["background_status"] = status
			state["background_scanned"] = scanned
			state["background_synced"] = synced
			state["background_error"] = error

		result = sync_all_emails(
			db, user, service,
			progress_callback=progress_callback,
			background_callback=background_callback,
		)
		
		# Always ensure AI classification is enabled and triggered on initial sync / login
		account = db.query(GmailAccount).filter(GmailAccount.user_id == user.id).first()
		if account:
			account.ai_classification_enabled = True
			db.commit()
			try:
				from app.services.llm_classification_service import (
					enqueue_classification_job,
					trigger_immediate_classification,
				)
				enqueue_classification_job(account.id, job_type="INITIAL")
				trigger_immediate_classification()
			except Exception as llm_exc:
				logger.warning("Failed to enqueue initial LLM classification: %s", llm_exc)

			# Trigger RAG auto-embedding for newly synced emails
			try:
				from app.services.rag_auto_indexer import ensure_user_emails_embedded
				ensure_user_emails_embedded(user.id)
			except Exception as rag_exc:
				logger.warning("Failed to trigger RAG auto-indexer after sync: %s", rag_exc)

		_sync_state[user_id_str].update({
			"status": "completed",
			"stage": "ai_labels",
			"ai_labels_status": "queued",
			"emails_found": result["emails_found"],
			"emails_synced": result["emails_synced"],
			"labels_synced": result["labels_synced"],
			"error": None,
		})
	except Exception as exc:
		logger.exception("Gmail sync failed for user %s", user_id_str)
		try:
			db.rollback()
			account = db.scalar(select(GmailAccount).where(GmailAccount.user_id == user_id))
			if account:
				account.sync_status = "ERROR"
				account.inbox_sync_heartbeat_at = None
				db.commit()
		except Exception:
			logger.exception("Unable to save Inbox sync failure for user %s", user_id_str)
		_sync_state[user_id_str] = {
			"status": "error",
			"stage": "error",
			"error": str(exc),
			"background_status": "error",
			"background_error": str(exc),
			"emails_found": _sync_state.get(user_id_str, {}).get("emails_found", 0),
			"emails_synced": _sync_state.get(user_id_str, {}).get("emails_synced", 0),
			"labels_synced": _sync_state.get(user_id_str, {}).get("labels_synced", 0),
		}
	finally:
		db.close()


@router.post("/sync")
def start_sync(
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""Start Gmail synchronization in the background."""
	if user.is_demo:
		raise HTTPException(status_code=403, detail="Gmail sync is unavailable in demo mode")
	user_id_str = str(user.id)

	account = db.scalar(select(GmailAccount).where(GmailAccount.user_id == user.id))
	if account is None:
		raise HTTPException(status_code=404, detail="No Gmail account linked")
	current = _sync_state.get(user_id_str, {})
	if _sync_lease_fresh(account.inbox_sync_heartbeat_at) and (
		current.get("status") == "syncing" or current.get("background_status") == "syncing"
	):
		return {"message": "Sync already in progress", **_get_sync_state(user_id_str, db)}
	now = datetime.now(timezone.utc)
	stale_before = now - _SYNC_LEASE
	reserved = db.execute(
		update(GmailAccount)
		.where(
			GmailAccount.id == account.id,
			or_(
				GmailAccount.sync_status.notin_(("INITIAL", "BACKFILL")),
				GmailAccount.inbox_sync_heartbeat_at < stale_before,
				GmailAccount.inbox_sync_heartbeat_at.is_(None),
			),
		)
		.values(sync_status="INITIAL", inbox_sync_heartbeat_at=now)
		.execution_options(synchronize_session=False)
	).rowcount
	db.commit()
	if not reserved:
		return {"message": "Sync already in progress", **_get_sync_state(user_id_str, db)}

	_sync_state[user_id_str] = {
		"status": "syncing",
		"stage": "starting",
		"emails_found": 0,
		"emails_synced": 0,
		"labels_synced": 0,
		"background_status": "idle",
		"background_scanned": 0,
		"background_synced": 0,
		"background_error": None,
		"error": None,
	}

	thread = threading.Thread(
		target=_run_sync_in_thread,
		args=(user_id_str, user.id),
		daemon=True,
	)
	thread.start()

	return {"message": "Sync started", **_get_sync_state(user_id_str)}


@router.get("/sync/status")
def sync_status(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
	"""Poll the current sync progress."""
	return _get_sync_state(str(user.id), db)


@router.get("/sync/stream")
async def sync_stream(user: User = Depends(get_current_user)):
	"""SSE endpoint for real-time sync progress updates."""
	user_id_str = str(user.id)

	async def event_generator():
		import json
		last_sent = None
		idle_count = 0

		while True:
			state = _get_sync_state(user_id_str)
			state_json = json.dumps(state)

			if state_json != last_sent:
				yield f"data: {state_json}\n\n"
				last_sent = state_json
				idle_count = 0

				if state.get("status") in ("completed", "error"):
					# Send one final event then close
					break
			else:
				idle_count += 1
				# After 5 minutes of no changes, close the stream
				if idle_count > 300:
					break

			await asyncio.sleep(1)

	return StreamingResponse(
		event_generator(),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			"Connection": "keep-alive",
			"X-Accel-Buffering": "no",
		},
	)

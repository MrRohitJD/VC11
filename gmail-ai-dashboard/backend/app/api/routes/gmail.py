"""Gmail sync routes — trigger sync, poll status, SSE stream for real-time progress."""

import asyncio
import logging
import threading
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.connection import get_db, SessionLocal
from app.models.email import User
from app.services.gmail_service import build_gmail_service_for_user, sync_all_emails

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/gmail", tags=["gmail"])

# In-memory sync state per user (keyed by str(user_id))
_sync_state: dict[str, dict] = {}


def _get_sync_state(user_id: str) -> dict:
	return _sync_state.get(user_id, {
		"status": "idle",
		"emails_found": 0,
		"emails_synced": 0,
		"labels_synced": 0,
		"stage": "",
		"error": None,
	})


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

		service = build_gmail_service_for_user(db, user)

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

		result = sync_all_emails(db, user, service, progress_callback=progress_callback)
		_sync_state[user_id_str] = {
			"status": "completed",
			"stage": "completed",
			"emails_found": result["emails_found"],
			"emails_synced": result["emails_synced"],
			"labels_synced": result["labels_synced"],
			"error": None,
		}
	except Exception as exc:
		logger.exception("Gmail sync failed for user %s", user_id_str)
		_sync_state[user_id_str] = {
			"status": "error",
			"stage": "error",
			"error": str(exc),
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
	user_id_str = str(user.id)

	current = _sync_state.get(user_id_str, {})
	if current.get("status") == "syncing":
		return {"message": "Sync already in progress", **_get_sync_state(user_id_str)}

	_sync_state[user_id_str] = {
		"status": "syncing",
		"stage": "starting",
		"emails_found": 0,
		"emails_synced": 0,
		"labels_synced": 0,
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
def sync_status(user: User = Depends(get_current_user)):
	"""Poll the current sync progress."""
	return _get_sync_state(str(user.id))


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

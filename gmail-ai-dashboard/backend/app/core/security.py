import base64
import hashlib
import hmac
import json
import threading
import time
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.connection import get_db
from app.models.email import User

_REVOKED_SESSIONS: dict[str, float] = {}
_REVOCATION_LOCK = threading.Lock()


def revoke_session(signature: str, expires_at: float) -> None:
	"""Add a session signature to the revocation list until its expiration."""
	now = time.time()
	with _REVOCATION_LOCK:
		expired = [sig for sig, exp in _REVOKED_SESSIONS.items() if exp <= now]
		for sig in expired:
			_REVOKED_SESSIONS.pop(sig, None)
		if expires_at > now:
			_REVOKED_SESSIONS[signature] = expires_at


def is_session_revoked(signature: str) -> bool:
	"""Check if a session signature has been revoked."""
	now = time.time()
	with _REVOCATION_LOCK:
		exp = _REVOKED_SESSIONS.get(signature)
		if exp is not None:
			if exp > now:
				return True
			_REVOKED_SESSIONS.pop(signature, None)
	return False


def revoke_request_session(request: Request) -> None:
	"""Extract and revoke the current session token from the request."""
	value = request.cookies.get("gmail_ai_session")
	if not value or "." not in value:
		return
	encoded, signature = value.rsplit(".", 1)
	try:
		padding = "=" * (-len(encoded) % 4)
		payload = json.loads(base64.urlsafe_b64decode(encoded + padding))
		exp = float(payload.get("exp", 0))
		revoke_session(signature, exp)
	except Exception:
		pass


def session_user_id(request: Request) -> UUID:
	value = request.cookies.get("gmail_ai_session")
	if not value or "." not in value:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
	encoded, signature = value.rsplit(".", 1)
	expected = hmac.new(settings.session_secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
	if not hmac.compare_digest(signature, expected):
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
	if is_session_revoked(signature):
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session has been revoked")
	try:
		padding = "=" * (-len(encoded) % 4)
		payload = json.loads(base64.urlsafe_b64decode(encoded + padding))
		user_id = UUID(payload["user_id"])
		if int(payload["exp"]) < time.time():
			raise ValueError
	except (KeyError, TypeError, ValueError, json.JSONDecodeError):
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")

	return user_id


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
	user_id = session_user_id(request)
	user = db.scalar(select(User).where(User.id == user_id))
	if user is None:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
	return user

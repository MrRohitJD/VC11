import base64
import hashlib
import hmac
import json
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.connection import get_db
from app.models.email import User


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
	value = request.cookies.get("gmail_ai_session")
	if not value or "." not in value:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
	encoded, signature = value.rsplit(".", 1)
	expected = hmac.new(settings.session_secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
	if not hmac.compare_digest(signature, expected):
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
	try:
		padding = "=" * (-len(encoded) % 4)
		payload = json.loads(base64.urlsafe_b64decode(encoded + padding))
		user_id = UUID(payload["user_id"])
		if int(payload["exp"]) < __import__("time").time():
			raise ValueError
	except (KeyError, TypeError, ValueError, json.JSONDecodeError):
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")

	user = db.scalar(select(User).where(User.id == user_id))
	if user is None:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
	return user

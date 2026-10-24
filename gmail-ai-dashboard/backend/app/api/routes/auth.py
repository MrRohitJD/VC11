import base64
import hashlib
import hmac
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import get_current_user
from app.database.connection import get_db
from app.database.crud import delete_user_gmail_data, serialize_user, upsert_google_user
from app.models.email import User
from app.services.chat_quota import set_demo_visitor_cookie
from app.services.demo_seed import DEMO_EMAIL, DEMO_GOOGLE_ID

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])
OAUTH_SCOPES = [
	"openid",
	"https://www.googleapis.com/auth/userinfo.email",
	"https://www.googleapis.com/auth/userinfo.profile",
	"https://www.googleapis.com/auth/gmail.readonly",
	"https://www.googleapis.com/auth/gmail.send",
]
SESSION_COOKIE = "gmail_ai_session"
STATE_COOKIE = "gmail_ai_oauth_state"
VERIFIER_COOKIE = "gmail_ai_oauth_verifier"


def _client_config() -> dict:
	if not settings.google_client_id or not settings.google_client_secret:
		raise HTTPException(status_code=500, detail="Google OAuth is not configured")
	return {"web": {"client_id": settings.google_client_id, "client_secret": settings.google_client_secret, "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token", "redirect_uris": [settings.oauth_redirect_uri]}}


def _flow(state: str | None = None, code_verifier: str | None = None) -> Flow:
	flow = Flow.from_client_config(_client_config(), scopes=OAUTH_SCOPES, state=state, code_verifier=code_verifier)
	flow.redirect_uri = settings.oauth_redirect_uri
	return flow


def _encode_session(user: User) -> str:
	lifetime = timedelta(days=1 if user.is_demo else 7)
	payload = json.dumps({"user_id": str(user.id), "exp": int((datetime.now(timezone.utc) + lifetime).timestamp())}, separators=(",", ":")).encode()
	encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
	signature = hmac.new(settings.session_secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
	return f"{encoded}.{signature}"


@router.get("/google")
def google_login(response: Response):
	state = secrets.token_urlsafe(32)
	flow = _flow(state)
	authorization_url, _ = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
	response = RedirectResponse(authorization_url, status_code=status.HTTP_302_FOUND)
	response.set_cookie(STATE_COOKIE, state, httponly=True, secure=settings.secure_cookies, samesite="lax", max_age=600)
	response.set_cookie(VERIFIER_COOKIE, flow.code_verifier or "", httponly=True, secure=settings.secure_cookies, samesite="lax", max_age=600)
	return response


@router.get("/callback")
def google_callback(request: Request, db: Session = Depends(get_db)):
	state = request.cookies.get(STATE_COOKIE)
	code_verifier = request.cookies.get(VERIFIER_COOKIE) or None
	code = request.query_params.get("code")
	if not state or not code or state != request.query_params.get("state"):
		logger.error("OAuth callback validation failed: state=%s, code=%s, state_match=%s", bool(state), bool(code), state == request.query_params.get("state"))
		raise HTTPException(status_code=400, detail="Invalid OAuth callback")

	flow = _flow(state, code_verifier)
	try:
		flow.fetch_token(code=code)
		credentials = flow.credentials
		claims = id_token.verify_oauth2_token(credentials.id_token, google_requests.Request(), settings.google_client_id)
	except Exception as exc:
		import traceback
		traceback.print_exc()
		print(f"OAUTH ERROR: {type(exc).__name__}: {exc}")
		raise HTTPException(status_code=400, detail=f"Google authentication failed: {type(exc).__name__}: {exc}") from exc

	if not claims.get("email_verified") or not claims.get("sub") or not claims.get("email"):
		raise HTTPException(status_code=400, detail="A verified Gmail account is required")
	if claims["email"].lower() == DEMO_EMAIL:
		raise HTTPException(status_code=409, detail="This address is reserved for the public demo")

	user = upsert_google_user(db, google_id=claims["sub"], email=claims["email"], name=claims.get("name"), picture=claims.get("picture"), access_token=credentials.token, refresh_token=credentials.refresh_token, token_expiry=credentials.expiry, scope=credentials.scopes and " ".join(credentials.scopes))

	# Automatically trigger RAG auto-embedding for any unindexed emails in background
	try:
		from app.services.rag_auto_indexer import ensure_user_emails_embedded
		ensure_user_emails_embedded(user.id)
	except Exception as auto_emb_exc:
		logger.warning("Failed to launch background auto-indexer for user %s: %s", user.id, auto_emb_exc)

	# Redirect to /sync so the frontend shows the sync loading screen
	response = RedirectResponse(f"{settings.frontend_url}/sync", status_code=status.HTTP_302_FOUND)
	response.set_cookie(SESSION_COOKIE, _encode_session(user), httponly=True, secure=settings.secure_cookies, samesite="lax", max_age=604800)
	response.delete_cookie(STATE_COOKIE)
	response.delete_cookie(VERIFIER_COOKIE)
	return response


@router.get("/me")
def current_user(user: User = Depends(get_current_user)):
	if not user.is_demo:
		try:
			from app.services.rag_auto_indexer import ensure_user_emails_embedded
			ensure_user_emails_embedded(user.id)
		except Exception:
			pass
	return serialize_user(user)


@router.post("/demo")
def enter_demo(request: Request, db: Session = Depends(get_db)):
	"""Open the seeded, read-only fictional mailbox without Google OAuth."""
	user = db.scalar(select(User).where(User.google_id == DEMO_GOOGLE_ID,
	                                    User.email == DEMO_EMAIL, User.is_demo.is_(True)))
	if user is None:
		raise HTTPException(status_code=503, detail="Demo mailbox has not been prepared")
	response = Response(content=json.dumps(serialize_user(user)), media_type="application/json")
	response.set_cookie(SESSION_COOKIE, _encode_session(user), httponly=True,
	                    secure=settings.secure_cookies, samesite="lax", max_age=86400)
	set_demo_visitor_cookie(request, response)
	return response


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request):
	"""End the user's session. Does NOT delete any stored data."""
	response = Response(status_code=status.HTTP_204_NO_CONTENT)
	response.delete_cookie(SESSION_COOKIE)
	return response


@router.delete("/erase", status_code=status.HTTP_204_NO_CONTENT)
def erase_all_data(
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""Erase stored mailbox data while preserving the profile and labels.

	This removes synchronized emails and all email-dependent records without
	deleting the ``users`` row, the user's labels, or the user's actual Gmail.
	"""
	delete_user_gmail_data(db, user)
	response = Response(status_code=status.HTTP_204_NO_CONTENT)
	response.delete_cookie(SESSION_COOKIE)
	return response

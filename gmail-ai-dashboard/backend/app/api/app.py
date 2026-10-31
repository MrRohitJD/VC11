from contextlib import asynccontextmanager
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.auth import google_callback, router as auth_router
from app.api.routes.emails import router as emails_router
from app.api.routes.gmail import router as gmail_router
from app.api.routes.llm import router as llm_router
from app.api.routes.rag import router as rag_router
from app.api.routes.developer import router as developer_router
from app.core.config import settings
from app.database.connection import init_db, SessionLocal, CentralSessionLocal
from app.core.security import get_current_user, session_user_id
from app.models.email import User
from app.services.developer_settings import (
    decrypt, database_url_context, llm_key_context, ensure_personal_identity,
    DatabaseRoutingMode, database_routing_mode_context, database_fallback_reason_context,
    set_personal_db_fallback, clear_personal_db_status, get_personal_db_status,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
	init_db()
	try:
		from app.services.llm_classification_service import start_global_ai_classifier_daemon
		start_global_ai_classifier_daemon()
	except Exception:
		pass
	try:
		from app.services.rag_auto_indexer import start_global_rag_auto_indexer_daemon
		start_global_rag_auto_indexer_daemon()
	except Exception:
		pass
	try:
		from app.services.email_scheduler import start_global_email_delivery_daemon
		start_global_email_delivery_daemon()
	except Exception:
		pass
	yield


app = FastAPI(title="Gmail AI Dashboard API", lifespan=lifespan)


@app.middleware("http")
async def route_developer_connections(request: Request, call_next):
    path = request.url.path
    central_paths = ("/api/auth/google", "/api/auth/callback", "/auth/google/callback",
                     "/api/auth/demo", "/api/auth/logout")
    if not path.startswith("/api/") or path.startswith(central_paths):
        return await call_next(request)
    try:
        user_id = session_user_id(request)
    except HTTPException:
        return await call_next(request)

    url = None
    provider_key = None
    mode = DatabaseRoutingMode.APP_DEFAULT.value
    fallback_reason = None

    with CentralSessionLocal() as db:
        owner = db.get(User, user_id)
        if owner is not None and not owner.is_demo:
            if owner.developer_llm_api_key:
                try:
                    decrypted_key = decrypt(owner.developer_llm_api_key)
                    provider_key = (owner.developer_llm_provider, decrypted_key)
                except Exception as exc:
                    logger.warning("Could not decrypt developer LLM key for user %s: %s", user_id, type(exc).__name__)

            if owner.developer_database_url:
                try:
                    url = decrypt(owner.developer_database_url)
                except Exception as exc:
                    logger.warning("Could not decrypt developer database URL for user %s: %s", user_id, type(exc).__name__)
                    url = None
                    fallback_reason = (
                        "Unable to connect to your personal database due to invalid credentials or a connection error. "
                        "VC11 has automatically switched to the default application database. "
                        "Your data operations will continue using the application database."
                    )
                    set_personal_db_fallback(user_id, fallback_reason)
                    mode = DatabaseRoutingMode.APP_FALLBACK.value

                if url:
                    # Check if personal database is in cooldown from prior failure
                    cached_fallback = get_personal_db_status(user_id)
                    if cached_fallback:
                        mode = DatabaseRoutingMode.APP_FALLBACK.value
                        fallback_reason = cached_fallback.get("reason")
                        url = None
                    else:
                        try:
                            ensure_personal_identity(owner, url)
                            mode = DatabaseRoutingMode.PERSONAL_ACTIVE.value
                            clear_personal_db_status(user_id)
                        except Exception as exc:
                            logger.error("Personal database connection failed for user %s: %s", user_id, type(exc).__name__)
                            # Automatic fallback to application database
                            url = None
                            mode = DatabaseRoutingMode.APP_FALLBACK.value
                            fallback_reason = (
                                "Unable to connect to your personal database due to invalid credentials or a connection error. "
                                "VC11 has automatically switched to the default application database. "
                                "Your data operations will continue using the application database."
                            )
                            set_personal_db_fallback(user_id, fallback_reason)

    # For settings routes, queries always execute against CentralSessionLocal
    if path in ("/api/developer", "/api/auth/me"):
        url = None

    db_token = database_url_context.set(url)
    key_token = llm_key_context.set(provider_key)
    mode_token = database_routing_mode_context.set(mode)
    reason_token = database_fallback_reason_context.set(fallback_reason)

    try:
        response = await call_next(request)
        response.headers["X-Database-Mode"] = mode
        if fallback_reason:
            response.headers["X-Database-Fallback-Reason"] = fallback_reason
        return response
    finally:
        database_url_context.reset(db_token)
        llm_key_context.reset(key_token)
        database_routing_mode_context.reset(mode_token)
        database_fallback_reason_context.reset(reason_token)
OLD_HEROKU_HOST = "ai-powered-gmail-0007703734bc.herokuapp.com"
allowed_origins = [settings.frontend_url]
if OLD_HEROKU_HOST not in settings.frontend_url:
    allowed_origins.append(f"https://{OLD_HEROKU_HOST}")

app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def redirect_old_domain(request: Request, call_next):
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").split(":")[0].lower()
    if host == OLD_HEROKU_HOST and request.url.path != "/health":
        target = f"{settings.frontend_url.rstrip('/')}{request.url.path}"
        if request.url.query:
            target += f"?{request.url.query}"
        return RedirectResponse(target, status_code=301)
    return await call_next(request)


@app.middleware("http")
async def protect_demo_mailbox(request: Request, call_next):
    allowed = {"/api/auth/demo", "/api/auth/logout", "/api/rag/chat", "/api/rag/chat/stream"}
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.url.path.startswith("/api/") and request.url.path not in allowed:
        with SessionLocal() as db:
            try:
                user = get_current_user(request, db)
            except HTTPException:
                user = None
            if user is not None and user.is_demo:
                return JSONResponse({"detail": "The sample mailbox is read only"}, status_code=403)
    return await call_next(request)
app.include_router(auth_router)
app.include_router(emails_router)
app.include_router(gmail_router)
app.include_router(llm_router)
app.include_router(rag_router)
app.include_router(developer_router)
app.add_api_route("/auth/google/callback", google_callback, methods=["GET"], include_in_schema=False)


@app.get("/health")
def health():
	return {"status": "ok"}


_frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if _frontend_dist.is_dir():
    if (_frontend_dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=_frontend_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path.startswith("api/") or path.startswith("auth/"):
            raise HTTPException(status_code=404)
        requested = (_frontend_dist / path).resolve()
        if requested.is_file() and requested.is_relative_to(_frontend_dist.resolve()):
            return FileResponse(requested)
        return FileResponse(_frontend_dist / "index.html")

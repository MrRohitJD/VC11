from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.auth import google_callback, router as auth_router
from app.api.routes.emails import router as emails_router
from app.api.routes.gmail import router as gmail_router
from app.api.routes.llm import router as llm_router
from app.api.routes.rag import router as rag_router
from app.core.config import settings
from app.database.connection import init_db, SessionLocal
from app.core.security import get_current_user


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
app.add_middleware(CORSMiddleware, allow_origins=[settings.frontend_url], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


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

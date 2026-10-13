from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.auth import google_callback, router as auth_router
from app.api.routes.emails import router as emails_router
from app.api.routes.gmail import router as gmail_router
from app.api.routes.llm import router as llm_router
from app.api.routes.rag import router as rag_router
from app.core.config import settings
from app.database.connection import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
	init_db()
	try:
		from app.services.llm_classification_service import start_global_ai_classifier_daemon
		start_global_ai_classifier_daemon()
	except Exception:
		pass
	yield


app = FastAPI(title="Gmail AI Dashboard API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[settings.frontend_url], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(auth_router)
app.include_router(emails_router)
app.include_router(gmail_router)
app.include_router(llm_router)
app.include_router(rag_router)
app.add_api_route("/auth/google/callback", google_callback, methods=["GET"], include_in_schema=False)


@app.get("/health")
def health():
	return {"status": "ok"}
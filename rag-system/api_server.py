"""Standalone FastAPI microservice for the Gmail RAG System.

Runs independently of the Gmail AI Dashboard backend.
Can be started on e.g. port 8001 via:
    uvicorn api_server:app --port 8001 --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import default_pipeline
from schemas import RAGQueryRequest, RAGQueryResponse
from vector_store import default_vector_store

app = FastAPI(
    title="Gmail RAG System API",
    description="Standalone Retrieval-Augmented Generation microservice for Gmail Mailbox intelligence.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Health check endpoint and vector store status."""
    return {
        "status": "healthy",
        "service": "rag-system",
        "indexed_chunks": default_vector_store.count(),
    }


@app.post("/rag/query", response_model=RAGQueryResponse)
def query_rag(request: RAGQueryRequest):
    """Execute a grounded natural language RAG query over indexed emails."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return default_pipeline.execute_query(request)


@app.post("/rag/index")
def index_email(payload: dict):
    """Index an email document into the RAG vector store."""
    email_id = payload.get("id") or payload.get("email_id")
    if not email_id:
        raise HTTPException(status_code=400, detail="Missing email_id")

    chunks_count = default_pipeline.index_email(
        email_id=str(email_id),
        subject=payload.get("subject", ""),
        body=payload.get("body", ""),
        from_email=payload.get("from_email", ""),
        date=payload.get("date", ""),
        to_email=payload.get("to_email", ""),
        sender_name=payload.get("sender_name", ""),
        ai_label=payload.get("ai_label", "OTHER"),
        labels=payload.get("labels", []),
    )
    return {"status": "indexed", "email_id": email_id, "chunks": chunks_count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

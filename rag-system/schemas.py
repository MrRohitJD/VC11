"""Pydantic schemas and data contracts for the Gmail RAG System."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class EmailMetadata(BaseModel):
    """Metadata attributes stored alongside each email chunk for filtering."""
    email_id: str
    gmail_message_id: Optional[str] = None
    thread_id: Optional[str] = None
    sender_email: Optional[str] = None
    sender_name: Optional[str] = None
    recipient_email: Optional[str] = None
    subject: Optional[str] = None
    date: Optional[str] = None
    timestamp: Optional[float] = None
    ai_label: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    has_attachment: bool = False


class EmailChunk(BaseModel):
    """A semantic chunk of an email document with vector embeddings."""
    chunk_id: str
    email_id: str
    chunk_index: int
    content: str
    metadata: EmailMetadata
    embedding: Optional[List[float]] = None


class SourceCitation(BaseModel):
    """A grounded citation pointing directly to an email source."""
    email_id: str
    subject: str
    sender: str
    date: str
    snippet: str
    ai_label: Optional[str] = None
    relevance_score: float = 0.0


class MetricItem(BaseModel):
    """Key metric or KPI extracted from the retrieved context."""
    label: str
    value: str
    detail: str


class EntityItem(BaseModel):
    """Structured entity extracted from retrieved emails."""
    name: str
    date: Optional[str] = None
    status: Optional[str] = None
    extra: Optional[str] = None


class RAGQueryRequest(BaseModel):
    """Natural language query request from the frontend assistant."""
    query: str
    user_id: Optional[str] = None
    top_k: int = 5
    filter_label: Optional[str] = None
    filter_date_start: Optional[str] = None
    filter_date_end: Optional[str] = None


class RAGQueryResponse(BaseModel):
    """Complete grounded answer with citations, metrics, and pipeline traces."""
    query: str
    answer: str
    response_type: str = "summary"  # summary, text, error
    metrics: List[MetricItem] = Field(default_factory=list)
    entities: List[EntityItem] = Field(default_factory=list)
    sources: List[SourceCitation] = Field(default_factory=list)
    pipeline_steps: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0

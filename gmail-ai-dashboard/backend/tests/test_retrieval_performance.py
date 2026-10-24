"""Focused contracts for provider routing, public activity, and fan-out."""

from __future__ import annotations

import threading
import time

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from app.rag.activity import activity_context
from app.rag.adaptive import plan_retrieval
from app.rag.retrieval import MultiQueryRetriever


def test_provider_routing_is_query_dependent():
    focused = plan_retrieval("What did the launch email say about the API?", mcp_enabled=True)
    assert focused.providers == ["vector"]
    assert "gmail_mcp" not in focused.providers

    fresh = plan_retrieval("Show all emails received last week", mcp_enabled=True)
    assert fresh.providers == ["bm25", "gmail_mcp"]
    assert "vector" not in fresh.providers

    disabled = plan_retrieval("Show all emails received last week", mcp_enabled=False)
    assert disabled.providers == ["bm25"]


def test_activity_events_are_public_and_ordered():
    with activity_context() as activity:
        activity.emit("retrieve_candidates", "Found candidates", provider="vector", secret="do not publish")
        activity.emit("answer_ready", "Answer ready", count=2)

    assert [event["event_id"] for event in activity.steps] == [1, 2]
    assert [event["phase"] for event in activity.steps] == ["retrieve_candidates", "answer_ready"]
    assert activity.steps[0]["provider"] == "vector"
    assert "secret" not in activity.steps[0]


class _ParallelRetriever(BaseRetriever):
    """Small retriever that records overlap without touching a DB session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    delay: float = Field(0.04)
    lock: object = Field(default_factory=threading.Lock, exclude=True)
    active: int = Field(0, exclude=True)
    max_active: int = Field(0, exclude=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return [Document(page_content=query, metadata={"email_id": query})]
        finally:
            with self.lock:
                self.active -= 1


def test_multi_query_retrieval_runs_branches_concurrently():
    base = _ParallelRetriever()
    retriever = MultiQueryRetriever(base_retriever=base, query_count=2)

    docs = retriever.invoke("invoice")

    assert len(docs) == 3
    assert base.max_active >= 2

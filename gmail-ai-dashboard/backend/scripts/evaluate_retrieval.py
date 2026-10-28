"""Compare legacy chunk-limited and distinct-email semantic retrieval.

Cases are JSONL records: {"query": "...", "relevant_email_ids": ["uuid", ...]}.
The script is read-only and prints aggregate rankings and latency, never email text.
Run from backend with an authorized mailbox UUID and the configured environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from uuid import UUID

from app.core.config import settings
from app.database.connection import SessionLocal
from app.rag.retrieval import PostgresSemanticRetriever


def ranking_metrics(found: list[str], relevant: set[str], k: int) -> tuple[float, float, float]:
    ranked = found[:k]
    hits = [index for index, email_id in enumerate(ranked, 1) if email_id in relevant]
    recall = len(hits) / len(relevant) if relevant else 0.0
    precision = len(hits) / k
    reciprocal_rank = 1 / hits[0] if hits else 0.0
    return recall, precision, reciprocal_rank


def evaluate(cases: list[dict], user_id: UUID, k: int, multiplier: int) -> dict:
    scores = []
    durations = []
    with SessionLocal() as db:
        retriever = PostgresSemanticRetriever(
            user_id=str(user_id), db_session=db, top_k=k,
            candidate_multiplier=multiplier,
        )
        for case in cases:
            started = perf_counter()
            documents = retriever.invoke(case["query"])
            durations.append((perf_counter() - started) * 1000)
            found = [str(doc.metadata["email_id"]) for doc in documents]
            scores.append(ranking_metrics(found, set(case["relevant_email_ids"]), k))
    ordered = sorted(durations)
    return {
        "cases": len(cases), "recall_at_k": round(mean(s[0] for s in scores), 4),
        "precision_at_k": round(mean(s[1] for s in scores), 4),
        "mrr_at_k": round(mean(s[2] for s in scores), 4),
        "retrieval_p50_ms": round(median(durations), 1),
        "retrieval_p95_ms": round(ordered[max(0, int(len(ordered) * .95 + .999) - 1)], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=UUID, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or any(not isinstance(case.get("query"), str)
                        or not isinstance(case.get("relevant_email_ids"), list)
                        or not case["relevant_email_ids"] for case in cases):
        parser.error("Cases must contain a query and at least one relevant_email_id")
    result = {
        "baseline_chunk_limit": evaluate(cases, args.user_id, args.top_k, 1),
        "distinct_email_candidates": evaluate(
            cases, args.user_id, args.top_k, settings.rag_vector_candidate_multiplier),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

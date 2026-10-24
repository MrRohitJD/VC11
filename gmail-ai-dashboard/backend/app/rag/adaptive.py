"""Deterministic retrieval planning and BM25 ranking over authorized emails."""
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from langchain_core.documents import Document
from app.rag.search_plan import exhaustive_request, date_only_request
from app.rag.query_constraints import received_date_bounds

STOPWORDS = set("a an the i my me you your we our find search show get list give please all every email emails mail mails message messages related about from to in on at for by of with and or is are was were be have has do does what when where who how can could would should summarize summarise summary brief overview incoming received today yesterday this current last past previous month months week weeks day days upcoming recent".split())
SYNONYMS = {
    "interview": {"assessment", "screening", "recruiter", "hiring", "interview", "interviews"},
    "bill": {"bill", "bills", "invoice", "invoices", "billing", "payment"},
    "security": {"security", "login", "password", "verification", "authentication"},
    "meeting": {"meeting", "meetings", "calendar", "invitation", "rescheduled"},
}


@dataclass
class RetrievalPlan:
    strategy: str
    exhaustive: bool
    date_only: bool
    multi_query: bool
    diverse: bool
    label: str
    date_start: str | None = None
    date_end: str | None = None
    # Public provider routing is deliberately descriptive.  It tells the UI
    # which bounded retrieval branches are active without exposing prompts or
    # model reasoning.
    providers: list[str] = field(default_factory=list)
    routing_reason: str = ""

    def public(self):
        return asdict(self)


def plan_retrieval(query, filters=None, use_multi_query=None, mode="similarity",
                   mcp_enabled=None):
    """Choose the retrieval strategy and the providers worth paying for.

    The local index is the fast default.  Live Gmail discovery is reserved for
    freshness-sensitive or broad Primary queries, where it provides current
    label/date membership.  Exact and structured requests stay local because
    their SQL/BM25 path is both faster and more precise than broad Gmail
    discovery.  Complex requests fan out to vector and BM25, with live Gmail
    added when freshness is part of the request.
    """
    if mcp_enabled is None:
        try:
            from app.core.config import settings
            mcp_enabled = bool(settings.gmail_mcp_enabled)
        except Exception:
            mcp_enabled = False

    query = query or ""
    exhaustive, date_only = exhaustive_request(query), date_only_request(query)
    bounds = received_date_bounds(query)
    diverse = bool(re.search(r"\b(?:overview|different topics|across|variety|themes)\b", query, re.I))
    complex_query = bool(re.search(r"\b(?:compare|versus|both|across)\b|\band\b.+\b(?:bills?|interviews?|meetings?|payments?)\b", query, re.I))
    multi = use_multi_query is True or (use_multi_query is None and complex_query)
    exact_filter = any((filters or {}).get(key) for key in
                       ("email_id", "gmail_message_id", "thread_id", "sender", "label"))
    freshness = bool(re.search(
        r"\b(?:today|yesterday|latest|newest|recent|recently|inbox|unread|upcoming|received|last|past)\b",
        query, re.I,
    ))
    if exhaustive or date_only:
        strategy, label, multi = "exhaustive_primary", "Read every matching Primary email", False
    elif any((filters or {}).get(key) for key in ("email_id", "gmail_message_id", "thread_id")):
        strategy, label, multi = "structured", "Look up exact messages and threads", False
    elif multi:
        strategy, label = "multi_query", "Search multiple query perspectives"
    elif diverse or mode == "mmr":
        strategy, label = "mmr", "Search for diverse email context"
    else:
        strategy, label = "vector", "Search the semantic email index"

    if exhaustive or date_only:
        providers = ["bm25"]
        routing_reason = "Broad or date-scoped Primary membership needs lexical coverage"
    elif exact_filter:
        providers = ["bm25", "vector"]
        routing_reason = "Explicit sender/label/identifier filters use precise local retrieval"
    elif multi or diverse:
        providers = ["vector", "bm25"]
        routing_reason = "Multi-topic requests benefit from dense and lexical recall"
    else:
        providers = ["vector"]
        routing_reason = "Focused requests use the indexed vector path first; lexical fallback is available on low confidence"

    if mcp_enabled and (exhaustive or date_only or freshness or multi or diverse):
        providers.append("gmail_mcp")
        routing_reason += "; live Gmail validates current Primary membership"

    return RetrievalPlan(strategy, exhaustive, date_only, multi, diverse or mode == "mmr", label,
                         bounds[0].isoformat() if bounds else None,
                         bounds[1].isoformat() if bounds else None,
                         providers, routing_reason)


def tokens(text):
    return re.findall(r"[\w@.\-]+", (text or "").lower())


def query_terms(query):
    terms = set(tokens(query)) - STOPWORDS
    terms = {t for t in terms if len(t) > 1 and not t.isdigit()}
    for root, variants in SYNONYMS.items():
        if root in terms or terms.intersection(variants):
            terms.update(variants)
    return terms


def bm25_rank(documents, query, limit=30):
    """BM25 with subject boosts. Documents must already satisfy tenant/scope filters."""
    terms = query_terms(query)
    if not terms or not documents:
        return []
    counts = [Counter(tokens(d.metadata.get("subject", "")) * 3 + tokens(d.page_content)) for d in documents]
    lengths = [sum(c.values()) for c in counts]
    average = max(sum(lengths) / len(lengths), 1)
    frequency = {term: sum(term in c for c in counts) for term in terms}
    ranked = []
    for doc, count, length in zip(documents, counts, lengths):
        score = 0.0
        for term in terms:
            tf = count.get(term, 0)
            if tf:
                idf = math.log(1 + (len(documents) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / average))
        if score:
            ranked.append((score, doc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    max_score = ranked[0][0] if ranked else 0.0
    normalized = []
    for raw_score, doc in ranked[:limit]:
        metadata = {**doc.metadata, "bm25_score": round(raw_score, 4), "retrieval_strategy": "bm25"}
        # Keep scores comparable with cosine similarity for the UI and for
        # confidence decisions. BM25 itself is unbounded, so normalize it
        # against the strongest candidate in this request.
        if metadata.get("score") is None:
            metadata["score"] = round(raw_score / max_score, 4) if max_score else 0.0
            metadata["score_source"] = "bm25_normalized"
        normalized.append(Document(page_content=doc.page_content, metadata=metadata))
    return normalized


def fuse_rankings(*rankings, limit=30):
    """Fuse independent rankings; repeated chunks in one list get one vote."""
    scores, documents = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, doc in enumerate(ranking, 1):
            eid = str(doc.metadata.get("email_id", ""))
            if not eid or eid in seen:
                continue
            seen.add(eid)
            documents.setdefault(eid, doc)
            scores[eid] = scores.get(eid, 0) + 1 / (60 + rank)
    ids = sorted(scores, key=scores.get, reverse=True)[:limit]
    fused = []
    for eid in ids:
        metadata = {**documents[eid].metadata, "fused_score": round(scores[eid], 4)}
        metadata.setdefault("score_source", metadata.get("retrieval_strategy", "rank_fusion"))
        fused.append(Document(page_content=documents[eid].page_content, metadata=metadata))
    return fused

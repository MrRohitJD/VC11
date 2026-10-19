"""Deterministic retrieval planning and BM25 ranking over authorized emails."""
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

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

    def public(self):
        return asdict(self)


def plan_retrieval(query, filters=None, use_multi_query=None, mode="similarity"):
    exhaustive, date_only = exhaustive_request(query), date_only_request(query)
    bounds = received_date_bounds(query)
    diverse = bool(re.search(r"\b(?:overview|different topics|across|variety|themes)\b", query, re.I))
    complex_query = bool(re.search(r"\b(?:compare|versus|both|across)\b|\band\b.+\b(?:bills?|interviews?|meetings?|payments?)\b", query, re.I))
    multi = use_multi_query is True or (use_multi_query is None and complex_query)
    if exhaustive or date_only:
        strategy, label, multi = "exhaustive_primary", "Read every matching Primary email", False
    elif any((filters or {}).get(key) for key in ("email_id", "gmail_message_id", "thread_id")):
        strategy, label, multi = "structured", "Look up exact messages and threads", False
    elif multi:
        strategy, label = "multi_query", "Search multiple query perspectives"
    elif diverse or mode == "mmr":
        strategy, label = "mmr", "Search for diverse email context"
    else:
        strategy, label = "hybrid_rrf", "Combine keyword and semantic search"
    return RetrievalPlan(strategy, exhaustive, date_only, multi, diverse or mode == "mmr", label,
                         bounds[0].isoformat() if bounds else None, bounds[1].isoformat() if bounds else None)


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
            ranked.append((score, Document(page_content=doc.page_content, metadata={**doc.metadata, "bm25_score": score, "retrieval_strategy": "bm25"})))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked[:limit]]


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
    return [Document(page_content=documents[eid].page_content,
                     metadata={**documents[eid].metadata, "fused_score": scores[eid]}) for eid in ids]

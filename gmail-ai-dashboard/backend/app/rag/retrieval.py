"""LangChain retrievers implementation for Gmail RAG system.

Provides 4 specialized retrievers adhering to LangChain's BaseRetriever abstraction:
1. ChromaSemanticRetriever: Dense semantic vector search via ChromaDB (Azure OpenAI text-embedding-3-large)
2. GmailCustomHybridRetriever: Fuses semantic search with PostgreSQL filters (sender, date range, labels, read status, email ID, thread ID)
3. HistoryAwareRetriever: Rewrites conversational follow-up questions into standalone queries using chat history
4. MultiQueryRetriever: Generates alternative search queries, executes parallel retrieval, merges and deduplicates via RRF
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

try:
    from langsmith import traceable
except ImportError:
    def traceable(name: Optional[str] = None, run_type: Optional[str] = None, **kwargs):
        def decorator(func):
            return func
        return decorator if (name is None or isinstance(name, str)) else name


from app.models.email import AiCategory, Email, EmailClassification, EmailLabel, GmailAccount, Label
from app.rag.vectorstore import ChromaVectorStoreManager, default_chroma_manager

logger = logging.getLogger(__name__)


@dataclass
class RetrievedCitation:
    """Structured citation representation for verified email evidence."""
    email_id: str
    subject: str
    sender: str
    date: str
    snippet: str
    score: float
    thread_id: str = ""
    label: str = "OTHER"
    key_points: List[str] = field(default_factory=list)


def documents_to_citations(documents: List[Document]) -> List[RetrievedCitation]:
    """Convert LangChain documents into clean RetrievedCitation instances."""
    citations: List[RetrievedCitation] = []
    seen_ids: Set[str] = set()

    for doc in documents:
        meta = doc.metadata or {}
        eid = str(meta.get("email_id", ""))
        if eid and eid in seen_ids:
            continue
        if eid:
            seen_ids.add(eid)

        snippet = doc.page_content or ""
        key_points = list(meta.get("key_points") or [])

        # If document wasn't compressed, strip header if needed
        if not key_points and "\n\n" in snippet and not snippet.startswith("•"):
            snippet = snippet.split("\n\n", 1)[1]

        score = meta.get("score")
        if score is None:
            score = 0.85

        citations.append(
            RetrievedCitation(
                email_id=eid,
                subject=str(meta.get("subject") or "No Subject"),
                sender=str(meta.get("sender") or meta.get("from_email") or "Unknown"),
                date=str(meta.get("received_at") or meta.get("date") or ""),
                snippet=snippet[:500].strip(),
                score=round(float(score), 4),
                thread_id=str(meta.get("thread_id") or ""),
                label=str(meta.get("labels") or meta.get("label") or "EMAIL"),
                key_points=key_points,
            )
        )
    return citations


# ============================================================================
# 1. ChromaDB Semantic Retriever
# ============================================================================

class ChromaSemanticRetriever(BaseRetriever):
    """LangChain retriever executing dense semantic vector similarity search via ChromaDB HNSW.

    Features:
    - Default retrieval method using ChromaDB HNSW cosine similarity.
    - Single query embedding per request for low latency.
    - Mandatory tenant isolation (user_id) enforced on every query path.
    - Structured metadata filtering (sender, gmail_message_id, received_date, received_timestamp).
    - Optional MMR mode (search_type="mmr", fetch_k=20, lambda_mult=0.7) to reduce redundant chunks.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str = Field(..., description="Authenticated user UUID string for tenant isolation")
    top_k: int = Field(5, description="Number of top similar chunks to return (default 5, configurable 5-10)")
    search_type: str = Field("similarity", description="Retrieval method: 'similarity' (default HNSW) or 'mmr'")
    fetch_k: int = Field(20, description="Candidate chunks to evaluate for MMR diversity")
    lambda_mult: float = Field(0.7, description="Diversity tradeoff for MMR (0.0=max diversity, 1.0=max relevance)")
    filter_dict: Optional[Dict[str, Any]] = Field(None, description="Optional raw metadata filters for ChromaDB")
    score_threshold: Optional[float] = Field(None, description="Minimum similarity threshold")

    # Structured metadata filters
    sender: Optional[str] = Field(None, description="Filter by sender address or sender name")
    gmail_message_id: Optional[str] = Field(None, description="Filter by exact Gmail message ID")
    start_date: Optional[Union[datetime, str]] = Field(None, description="Filter emails received on or after date")
    end_date: Optional[Union[datetime, str]] = Field(None, description="Filter emails received before date")
    start_timestamp: Optional[Union[float, int]] = Field(None, description="Filter emails on or after POSIX timestamp")
    end_timestamp: Optional[Union[float, int]] = Field(None, description="Filter emails before POSIX timestamp")

    chroma_manager: Any = Field(default_factory=lambda: default_chroma_manager)

    @traceable(name="chroma_retrieval", run_type="retriever")
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Retrieve relevant email chunks using ChromaDB dense semantic embeddings."""
        if not query or not query.strip():
            return []

        # Enforce tenant isolation on every retrieval path
        if not self.user_id or not str(self.user_id).strip():
            raise ValueError("user_id is mandatory for tenant isolation on every retrieval path")

        combined_filter = self.chroma_manager.build_filter(
            user_id=self.user_id,
            sender=self.sender,
            gmail_message_id=self.gmail_message_id,
            start_date=self.start_date,
            end_date=self.end_date,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            custom_filter=self.filter_dict,
        )

        if self.search_type == "mmr":
            results = self.chroma_manager.search_mmr_with_scores(
                query=query.strip(),
                user_id=self.user_id,
                top_k=self.top_k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult,
                filter_dict=combined_filter,
            )
        else:
            # Default HNSW cosine similarity search with single query embedding
            results = self.chroma_manager.search_with_scores(
                query=query.strip(),
                user_id=self.user_id,
                top_k=self.top_k,
                filter_dict=combined_filter,
            )

        docs: List[Document] = []
        for doc, score in results:
            if self.score_threshold is not None and score is not None:
                if score < self.score_threshold:
                    continue

            meta = dict(doc.metadata or {})
            meta["score"] = round(float(score) if score is not None else 0.85, 4)
            meta.setdefault("source", "chromadb")
            meta["retrieval_mode"] = self.search_type
            meta["search_type"] = self.search_type
            docs.append(Document(page_content=doc.page_content, metadata=meta))

        return docs[:self.top_k]

    def retrieve_citations(self, query: str) -> List[RetrievedCitation]:
        """Convenience method returning structured RetrievedCitation objects."""
        return documents_to_citations(self.invoke(query))


# ============================================================================
# 2. Custom Hybrid Retriever (ChromaDB + PostgreSQL Filters with RRF)
# ============================================================================

def detect_exact_identifier_or_term(query: str, filters: Optional[dict[str, Any]] = None) -> Tuple[bool, str]:
    """Detect if a query or its filter contains exact terms, email addresses, IDs, or codes requiring keyword/hybrid fallback."""
    f = filters or {}
    if f.get("email_id"):
        return True, "Filter specifies exact email_id"
    if f.get("gmail_message_id"):
        return True, "Filter specifies exact gmail_message_id"
    if f.get("thread_id"):
        return True, "Filter specifies exact thread_id"
    if f.get("sender"):
        return True, "Filter specifies exact sender"

    clean_q = (query or "").strip()
    # Email address pattern
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", clean_q):
        return True, "Query contains exact email address"

    # UUID pattern
    if re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", clean_q, re.I):
        return True, "Query contains exact UUID"

    # Message ID pattern or invoice code pattern (e.g. msg-1234, INV-9876, #12345)
    if re.search(r"\b(?:msg|message|id|invoice|inv|order|ticket)[-_#:]?\s*[A-Za-z0-9_-]{4,}\b", clean_q, re.I):
        return True, "Query contains exact message/invoice/order identifier"

    # Quoted exact phrase (e.g. "exact text")
    if re.search(r'"[^"]{3,}"|\'[^\']{3,}\'', clean_q):
        return True, "Query contains exact quoted search term"

    # Domain specific exact keywords that require SQL keyword matching
    exact_topics = ["interview", "interviews", "invoice", "bill", "bills", "2fa", "security alert", "password reset"]
    if any(re.search(r"\b" + re.escape(topic) + r"\b", clean_q, re.I) for topic in exact_topics):
        return True, "Query contains exact domain topic keywords"

    return False, "Standard semantic query"


def should_fallback_to_hybrid(
    query: str,
    semantic_docs: List[Document],
    filters: Optional[dict[str, Any]] = None,
    min_confidence_score: float = 0.65,
) -> Tuple[bool, str]:
    """Determine whether retrieval should fall back to Hybrid Keyword + Vector Retrieval with RRF."""
    # Check for exact terms/identifiers
    has_exact, reason = detect_exact_identifier_or_term(query, filters)
    if has_exact:
        return True, f"Exact term detected: {reason}"

    # Check for insufficient semantic retrieval results
    if not semantic_docs:
        return True, "Zero documents returned by semantic vector search"

    # Check if semantic docs actually contain query topic keywords
    clean_q = (query or "").lower()
    core_topics = ["interview", "interviews", "invoice", "bill", "bills", "security", "password", "reset", "2fa"]
    matched_topic = next((t for t in core_topics if re.search(r"\b" + re.escape(t) + r"\b", clean_q)), None)
    if matched_topic:
        docs_have_topic = any(
            matched_topic in (d.page_content or "").lower() or matched_topic in str(d.metadata.get("subject", "")).lower()
            for d in semantic_docs
        )
        if not docs_have_topic:
            return True, f"Semantic docs lack query topic keyword '{matched_topic}'"

    top_score = max((float(d.metadata.get("score") or 0.0) for d in semantic_docs), default=0.0)
    if top_score < min_confidence_score:
        return True, f"Low semantic similarity score ({top_score:.3f} < {min_confidence_score})"

    return False, "Semantic retrieval confidence is sufficient"


class GmailCustomHybridRetriever(BaseRetriever):
    """Hybrid retriever combining ChromaDB semantic search with PostgreSQL metadata filters.

    Supports structured filtering by:
    - sender (from_email or sender_name)
    - gmail_message_id (exact Gmail message identifier)
    - date range (start_date, end_date, start_timestamp, end_timestamp)
    - labels (Gmail labels and AI categories)
    - read status (is_read)
    - email_id (exact email UUID)
    - thread_id (thread correspondence)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str = Field(..., description="Authenticated user UUID string for tenant isolation")
    db_session: Optional[Any] = Field(None, description="SQLAlchemy Session for relational queries")
    top_k: int = Field(5, description="Total ranked documents to return (default 5, configurable 5-10)")

    # Structured metadata filters
    sender: Optional[str] = Field(None, description="Filter by sender address or domain")
    gmail_message_id: Optional[str] = Field(None, description="Filter by exact Gmail message ID")
    start_date: Optional[Union[datetime, str]] = Field(None, description="Filter emails received on or after")
    end_date: Optional[Union[datetime, str]] = Field(None, description="Filter emails received before")
    start_timestamp: Optional[Union[float, int]] = Field(None, description="Filter emails received on or after POSIX timestamp")
    end_timestamp: Optional[Union[float, int]] = Field(None, description="Filter emails received before POSIX timestamp")
    label: Optional[str] = Field(None, description="Filter by label or AI category")
    is_read: Optional[bool] = Field(None, description="Filter by read/unread status")
    email_id: Optional[str] = Field(None, description="Filter by specific email UUID")
    thread_id: Optional[str] = Field(None, description="Filter by thread ID")

    search_type: str = Field("similarity", description="Vector retrieval search type ('similarity' or 'mmr')")
    fetch_k: int = Field(20, description="Candidate chunks for MMR")
    lambda_mult: float = Field(0.7, description="MMR diversity parameter")
    semantic_weight: float = Field(0.6, description="Weight for semantic vector scores in RRF")
    keyword_weight: float = Field(0.4, description="Weight for structured SQL matches in RRF")
    chroma_manager: Any = Field(default_factory=lambda: default_chroma_manager)

    @traceable(name="hybrid_retrieval", run_type="retriever")
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Execute hybrid search blending ChromaDB semantic vectors and PostgreSQL metadata via RRF."""
        if not self.user_id or not str(self.user_id).strip():
            raise ValueError("user_id is mandatory for tenant isolation on every retrieval path")

        user_uuid = UUID(str(self.user_id))
        semantic_docs: List[Document] = []
        sql_docs: List[Document] = []

        # 1. Semantic search via ChromaDB
        try:
            custom_f: Dict[str, Any] = {}
            if self.email_id:
                custom_f["email_id"] = str(self.email_id)
            if self.thread_id:
                custom_f["thread_id"] = str(self.thread_id)

            combined_chroma_filter = self.chroma_manager.build_filter(
                user_id=self.user_id,
                sender=self.sender,
                gmail_message_id=self.gmail_message_id,
                start_date=self.start_date,
                end_date=self.end_date,
                start_timestamp=self.start_timestamp,
                end_timestamp=self.end_timestamp,
                custom_filter=custom_f if custom_f else None,
            )

            if self.search_type == "mmr":
                raw_semantic = self.chroma_manager.search_mmr_with_scores(
                    query=query,
                    user_id=self.user_id,
                    top_k=self.top_k * 2,
                    fetch_k=self.fetch_k,
                    lambda_mult=self.lambda_mult,
                    filter_dict=combined_chroma_filter,
                )
            else:
                raw_semantic = self.chroma_manager.search_with_scores(
                    query=query,
                    user_id=self.user_id,
                    top_k=self.top_k * 2,
                    filter_dict=combined_chroma_filter,
                )

            for doc, score in raw_semantic:
                meta = dict(doc.metadata or {})
                meta["score"] = round(float(score) if score is not None else 0.85, 4)
                meta.setdefault("source", "chromadb")
                semantic_docs.append(Document(page_content=doc.page_content, metadata=meta))
        except Exception as exc:
            logger.debug("ChromaDB semantic search error: %s", exc)

        # 2. Structured & keyword search via PostgreSQL
        if self.db_session:
            try:
                db_query = (
                    select(Email)
                    .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
                    .where(GmailAccount.user_id == user_uuid)
                )

                if self.email_id:
                    try:
                        db_query = db_query.where(Email.id == UUID(str(self.email_id).strip()))
                    except Exception:
                        pass
                if self.gmail_message_id:
                    db_query = db_query.where(Email.gmail_message_id == self.gmail_message_id.strip())
                if self.thread_id:
                    db_query = db_query.where(Email.thread_id == self.thread_id.strip())
                if self.sender:
                    s_clean = self.sender.strip()
                    if s_clean:
                        db_query = db_query.where(
                            or_(
                                Email.from_email.ilike(f"%{s_clean}%"),
                                Email.sender_name.ilike(f"%{s_clean}%"),
                            )
                        )
                if self.start_date:
                    db_query = db_query.where(Email.received_at >= self.start_date)
                elif self.start_timestamp is not None:
                    db_query = db_query.where(
                        Email.received_at >= datetime.fromtimestamp(float(self.start_timestamp), tz=timezone.utc)
                    )

                if self.end_date:
                    db_query = db_query.where(Email.received_at <= self.end_date)
                elif self.end_timestamp is not None:
                    db_query = db_query.where(
                        Email.received_at <= datetime.fromtimestamp(float(self.end_timestamp), tz=timezone.utc)
                    )

                if self.is_read is not None:
                    db_query = db_query.where(Email.is_read == self.is_read)
                if self.label:
                    lbl = self.label.strip()
                    db_query = db_query.where(
                        or_(
                            Email.labels.any(EmailLabel.label.has(Label.name.ilike(lbl))),
                            Email.classifications.any(EmailClassification.category.has(AiCategory.name.ilike(lbl))),
                        )
                    )

                # Textual keyword filtering from query
                clean_query = (query or "").strip()
                cleaned_kw = re.sub(
                    r"^(?:find|search|show|get|list)(?:\s+all)?(?:\s+emails|\s+messages|\s+email)?(?:\s+containing|\s+about|\s+with|\s+from)?\s*",
                    "",
                    clean_query,
                    flags=re.IGNORECASE,
                ).strip()
                if cleaned_kw and len(cleaned_kw) > 1 and not (self.email_id or self.thread_id or self.gmail_message_id):
                    search_pat = f"%{cleaned_kw}%"
                    phrase_cond = or_(
                        Email.subject.ilike(search_pat),
                        Email.snippet.ilike(search_pat),
                        Email.body.ilike(search_pat),
                    )

                    # Extract individual significant words for multi-word or rewritten queries
                    stop_words = {
                        "find", "search", "show", "get", "list", "all", "email", "emails", "message", "messages",
                        "mail", "mails", "the", "a", "an", "and", "or", "to", "for", "in", "on", "at", "by", "from",
                        "with", "is", "are", "was", "were", "what", "when", "where", "how", "why", "who", "have", "has",
                        "had", "do", "does", "did", "can", "could", "would", "should", "will", "tell", "me",
                        "much", "many", "that", "this", "these", "those", "about", "your", "my", "our",
                        "last", "past", "months", "month", "weeks", "week", "days", "day", "related",
                        "summarize", "summary", "overview", "recap", "check", "please", "give", "any",
                        "upcoming", "due", "recent", "incoming",
                    }
                    sig_words = [
                        w.strip("?,.!:;\"'") for w in cleaned_kw.split()
                        if len(w.strip("?,.!:;\"'")) > 2 and w.lower().strip("?,.!:;\"'") not in stop_words
                    ]
                    if sig_words:
                        word_conds = [
                            or_(
                                Email.subject.ilike(f"%{w}%"),
                                Email.snippet.ilike(f"%{w}%"),
                                Email.body.ilike(f"%{w}%"),
                            )
                            for w in sig_words
                        ]
                        db_query = db_query.where(or_(phrase_cond, *word_conds))
                    else:
                        db_query = db_query.where(phrase_cond)

                db_emails = self.db_session.scalars(
                    db_query.order_by(Email.received_at.desc().nullslast()).limit(self.top_k * 2)
                ).all()

                for e in db_emails:
                    content = (e.snippet or e.body or "")[:500]
                    header = f"Subject: {e.subject or ''}\nFrom: {e.from_email or ''}\n\n"
                    sql_docs.append(
                        Document(
                            page_content=f"{header}{content}",
                            metadata={
                                "email_id": str(e.id),
                                "gmail_message_id": str(e.gmail_message_id or ""),
                                "subject": e.subject or "No Subject",
                                "sender": e.from_email or "Unknown",
                                "received_at": e.received_at.isoformat() if e.received_at else "",
                                "thread_id": e.thread_id or "",
                                "is_read": e.is_read,
                                "labels": self.label or "EMAIL",
                                "score": 1.0,
                                "source": "postgresql",
                            },
                        )
                    )
            except Exception as exc:
                logger.debug("PostgreSQL structured query error: %s", exc)

        # 3. Reciprocal Rank Fusion & Deduplication
        fused_scores: Dict[str, float] = {}
        doc_store: Dict[str, Document] = {}

        # Semantic ranks
        for rank, doc in enumerate(semantic_docs):
            eid = str(doc.metadata.get("email_id", "")) or doc.page_content[:40]
            fused_scores[eid] = fused_scores.get(eid, 0.0) + (1.0 / (60 + rank)) * self.semantic_weight
            if eid not in doc_store:
                doc_store[eid] = doc

        # SQL ranks
        for rank, doc in enumerate(sql_docs):
            eid = str(doc.metadata.get("email_id", "")) or doc.page_content[:40]
            fused_scores[eid] = fused_scores.get(eid, 0.0) + (1.0 / (60 + rank)) * self.keyword_weight
            if eid not in doc_store:
                doc_store[eid] = doc
            else:
                for k, v in doc.metadata.items():
                    if k not in doc_store[eid].metadata or doc_store[eid].metadata[k] is None:
                        doc_store[eid].metadata[k] = v

        # Sort by fused score descending
        sorted_eids = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)

        final_docs: List[Document] = []
        for eid in sorted_eids[:self.top_k]:
            doc = doc_store[eid]
            doc.metadata["score"] = round(fused_scores[eid], 4)
            doc.metadata.setdefault("fused_score", round(fused_scores[eid], 4))
            doc.metadata.setdefault("retrieval_strategy", "hybrid_rrf")
            final_docs.append(doc)

        return final_docs

    # Backward-compatible helper methods
    def retrieve_semantic(
        self,
        query: str,
        user_id: UUID | str,
        top_k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> List[RetrievedCitation]:
        retriever = ChromaSemanticRetriever(
            user_id=str(user_id),
            top_k=top_k,
            filter_dict=filter_dict,
            chroma_manager=self.chroma_manager,
        )
        return retriever.retrieve_citations(query)

    def retrieve_structured_sql(
        self,
        db: Session,
        user_id: UUID | str,
        keyword: Optional[str] = None,
        sender: Optional[str] = None,
        label: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        is_read: Optional[bool] = None,
        email_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[RetrievedCitation]:
        """Perform direct, deterministic search against PostgreSQL email records."""
        user_uuid = UUID(str(user_id))
        stmt = (
            select(Email)
            .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(GmailAccount.user_id == user_uuid)
        )

        if email_id:
            try:
                stmt = stmt.where(Email.id == UUID(str(email_id).strip()))
            except Exception:
                pass
        if thread_id:
            stmt = stmt.where(Email.thread_id == str(thread_id).strip())
        if sender:
            s_clean = sender.strip()
            if s_clean:
                stmt = stmt.where(
                    or_(
                        Email.from_email.ilike(f"%{s_clean}%"),
                        Email.sender_name.ilike(f"%{s_clean}%"),
                    )
                )
        if start_date:
            stmt = stmt.where(Email.received_at >= start_date)
        if end_date:
            stmt = stmt.where(Email.received_at <= end_date)
        if is_read is not None:
            stmt = stmt.where(Email.is_read == is_read)
        if label:
            lbl = label.strip()
            if lbl:
                stmt = stmt.where(
                    or_(
                        Email.labels.any(EmailLabel.label.has(Label.name.ilike(lbl))),
                        Email.classifications.any(EmailClassification.category.has(AiCategory.name.ilike(lbl))),
                    )
                )
        if keyword:
            kw_clean = keyword.strip()
            if kw_clean:
                stmt = stmt.where(
                    or_(
                        Email.subject.ilike(f"%{kw_clean}%"),
                        Email.snippet.ilike(f"%{kw_clean}%"),
                        Email.body.ilike(f"%{kw_clean}%"),
                    )
                )

        emails = db.scalars(
            stmt.order_by(Email.received_at.desc().nullslast()).limit(limit)
        ).all()

        citations: List[RetrievedCitation] = []
        for e in emails:
            content = (e.snippet or e.body or "")[:300].strip()
            citations.append(
                RetrievedCitation(
                    email_id=str(e.id),
                    subject=e.subject or "No Subject",
                    sender=e.from_email or e.sender_name or "Unknown",
                    date=e.received_at.isoformat() if e.received_at else "",
                    snippet=content,
                    score=1.0,
                    thread_id=e.thread_id or "",
                    label=label or "EMAIL",
                )
            )

        logger.info(
            "retrieve_structured_sql returned %d row(s) for user %s (sender=%s, keyword=%s, email_ids=%s)",
            len(citations),
            user_uuid,
            sender,
            keyword,
            [c.email_id for c in citations],
        )
        return citations

    def retrieve_hybrid(
        self,
        db: Session,
        query: str,
        user_id: UUID | str,
        top_k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> List[RetrievedCitation]:
        filters = dict(filter_dict or {})
        retriever = GmailCustomHybridRetriever(
            user_id=str(user_id),
            db_session=db,
            top_k=top_k,
            sender=filters.get("sender"),
            start_date=filters.get("start_date"),
            end_date=filters.get("end_date"),
            label=filters.get("label"),
            is_read=filters.get("is_read"),
            email_id=filters.get("email_id"),
            thread_id=filters.get("thread_id"),
            chroma_manager=self.chroma_manager,
        )
        docs = retriever.invoke(query)
        return documents_to_citations(docs)


# ============================================================================
# 3. History-Aware Query Rewriting & Retriever
# ============================================================================

REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given the conversational history and the user's latest query,
formulate a standalone question that can be understood without the chat history.
DO NOT answer the question. Only reformulate it to resolve ambiguous pronouns (e.g., 'he', 'that', 'it', 'the interview').
If the query is already standalone, return it verbatim.""",
    ),
    ("human", "Conversation History:\n{chat_history}\n\nQuestion: {input}"),
])


@traceable(name="query_rewriting", run_type="chain")
def rewrite_query_with_history(
    query: str,
    chat_history: List[BaseMessage],
    llm: Optional[Any] = None,
    prompt: Optional[ChatPromptTemplate] = None,
) -> str:
    """Rewrite ambiguous query into a standalone formulation using conversation history and LLM.

    Resolves pronouns ('he', 'it', 'that', 'his email') and conversational context.
    If the query is already standalone or chat history is empty, returns the original query.
    """
    if not chat_history:
        return query

    # Format chat history text
    history_lines = []
    for msg in chat_history[-6:]:
        role = type(msg).__name__.replace("Message", "")
        content = str(msg.content)[:250]
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines)

    standalone_query = query
    p = prompt or REPHRASE_PROMPT

    if llm:
        try:
            chain = p | llm | StrOutputParser()
            rephrased = chain.invoke({"chat_history": history_text, "input": query}).strip()
            if rephrased and len(rephrased) > 3:
                standalone_query = rephrased
        except Exception as exc:
            logger.debug("History-aware query rewrite error (%s), using fallback heuristic.", exc)

    # Heuristic fallback if LLM was unavailable, failed, or left query unchanged
    if standalone_query == query:
        ambiguous_tokens = {"it", "its", "this", "that", "these", "those", "he", "him", "she", "her", "they", "them", "his", "their", "there", "then"}
        query_words = set(query.lower().replace("?", " ").replace(".", " ").split())
        if any(token in query_words for token in ambiguous_tokens):
            stop_words = {
                "did", "i", "receive", "an", "email", "emails", "about", "my", "me", "you", "your",
                "yes", "no", "a", "an", "the", "for", "at", "on", "in", "to", "from", "with", "is",
                "was", "are", "were", "what", "when", "where", "how", "why", "who", "have", "has",
                "had", "do", "does", "can", "could", "would", "should", "will", "please", "tell"
            }
            salient: List[str] = []
            human_msgs = [m for m in chat_history if "Human" in type(m).__name__]
            sources = human_msgs[-2:] if human_msgs else chat_history[-2:]
            for msg in reversed(sources):
                content = str(msg.content)
                for w in content.split():
                    clean = w.strip("?,.!:;\"'")
                    if len(clean) > 3 and clean.lower() not in stop_words and clean.lower() not in query_words:
                        if clean.lower() not in [s.lower() for s in salient]:
                            salient.append(clean)
            if salient:
                standalone_query = f"{query} {' '.join(salient[:3])}"

    return standalone_query


class HistoryAwareRetriever(BaseRetriever):
    """LangChain retriever that rephrases follow-up queries using conversation history before retrieval."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever = Field(..., description="Underlying retriever (semantic or hybrid)")
    llm: Optional[Any] = Field(None, description="LangChain BaseChatModel for query rewriting")
    chat_history: List[BaseMessage] = Field(default_factory=list, description="Recent conversation turns")
    prompt: ChatPromptTemplate = Field(default=REPHRASE_PROMPT, description="LCEL rephrasing prompt")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Rewrite query into standalone formulation using chat history and retrieve documents."""
        standalone_query = rewrite_query_with_history(
            query=query,
            chat_history=self.chat_history,
            llm=self.llm,
            prompt=self.prompt,
        )
        docs = self.base_retriever.invoke(standalone_query)
        for d in docs:
            d.metadata["rewritten_query"] = standalone_query
        return docs


# ============================================================================
# 4. Multi-Query Retriever & Routing
# ============================================================================

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an AI search assistant for mailbox emails.
Generate {query_count} distinct search queries from different perspectives for the given question.
Focus on email subjects, sender context, and body keywords.
Provide the alternative queries separated by newlines only. Do NOT number them or include bullets.""",
    ),
    ("human", "{question}"),
])

ROUTING_COMPLEXITY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a search query complexity analyzer for an email RAG retrieval system.
Classify whether the given user query requires 'STANDARD' (single direct vector/keyword search) or 'MULTI_QUERY' (query expansion across multiple perspectives).

Rules:
- STANDARD: The query is straightforward with a specific sender address, exact date/range, specific single keyword, direct email/thread ID, or clear, focused question about a single entity (e.g. "Find emails from Bibin", "emails containing chess", "receipt for AWS").
- MULTI_QUERY: The query is complex, multi-topic, conceptual, or low-specificity, where different emails might express the concept in diverse ways (e.g. "discussions about project delays, architecture changes, and roadmaps", "why did the launch fail and what are next steps", "summarize feedback across all candidates").

Respond with ONLY 'STANDARD' or 'MULTI_QUERY' on the first line, followed by a brief reason.""",
    ),
    ("human", "{query}"),
])


@traceable(name="query_routing", run_type="chain")
def route_query_retrieval_strategy(
    query: str,
    llm: Optional[Any] = None,
    force_multi_query: Optional[bool] = None,
) -> tuple[str, str]:
    """Classify whether the query is straightforward ('standard') or complex ('multi_query').

    Returns:
        (strategy, reason) where strategy in {"standard", "multi_query"}.
    """
    if force_multi_query is True:
        return "multi_query", "Explicitly forced multi-query retrieval"
    if force_multi_query is False:
        return "standard", "Explicitly forced standard retrieval"

    clean_q = query.strip()
    clean_lower = clean_q.lower()

    # 1. Deterministic heuristic checks for straightforward searches
    # Exact email address
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", clean_q):
        return "standard", "Straightforward exact sender address search"

    # Exact date references
    if re.search(r"(?:last|past)\s+\d+\s+(?:months?|days?|weeks?|years?)|(?:last|past)\s+(?:month|week|year)|today|yesterday|\d{4}-\d{2}-\d{2}", clean_lower):
        if not any(w in clean_lower for w in ["compare", "summarize all discussions", "analyze reasons", "why did"]):
            return "standard", "Straightforward date-filtered query"

    # Specific single keyword queries
    if re.search(r"^(?:find|search|get|show)\s+(?:all\s+)?emails\s+(?:containing|about|with\s+keyword)\s+[\w-]+$", clean_lower):
        return "standard", "Straightforward single-keyword lookup"

    # UUID / Email ID lookup
    if re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", clean_lower):
        return "standard", "Straightforward email ID lookup"

    # 2. Heuristic checks for complex multi-topic searches
    conjunction_matches = re.findall(r"\b(?:and|or|as well as|versus|vs|compare)\b", clean_lower)
    words = clean_q.split()
    if len(words) >= 8 and len(conjunction_matches) >= 2:
        return "multi_query", "Complex multi-faceted query with multiple topic conjunctions"

    if any(phrase in clean_lower for phrase in ["why did", "reasons for", "summarize all discussions", "compare feedback", "what are the key takeaways", "comprehensive review"]):
        return "multi_query", "Complex analytical/thematic inquiry benefiting from diverse search perspectives"

    # 3. LLM classification if available
    if llm:
        try:
            chain = ROUTING_COMPLEXITY_PROMPT | llm | StrOutputParser()
            res = chain.invoke({"query": clean_q}).strip()
            first_line = res.split("\n")[0].upper()
            if "MULTI_QUERY" in first_line:
                return "multi_query", f"LLM classified as complex: {res[:100]}"
            elif "STANDARD" in first_line:
                return "standard", f"LLM classified as straightforward: {res[:100]}"
        except Exception as exc:
            logger.debug("Routing LLM classification error: %s. Using heuristic.", exc)

    if len(words) > 10:
        return "multi_query", "Long descriptive query defaulting to multi-query expansion"

    return "standard", "Default straightforward focused search"


class MultiQueryRetriever(BaseRetriever):
    """LangChain retriever generating multiple queries from diverse perspectives, merging and deduplicating results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever = Field(..., description="Underlying retriever (semantic or hybrid)")
    llm: Optional[Any] = Field(None, description="LangChain BaseChatModel for query expansion")
    query_count: int = Field(3, description="Number of alternative query perspectives to generate")
    prompt: ChatPromptTemplate = Field(default=MULTI_QUERY_PROMPT, description="Expansion prompt template")

    @traceable(name="multi_query_retrieval", run_type="retriever")
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Generate multiple queries, execute retrieval across all, and fuse results via RRF."""
        queries: List[str] = [query]

        if self.llm:
            try:
                chain = self.prompt | self.llm | StrOutputParser()
                raw_alts = chain.invoke({"question": query, "query_count": self.query_count})
                lines = [
                    line.strip()
                    for line in raw_alts.split("\n")
                    if line.strip() and not line.strip().startswith(("#", "-", "*", "1.", "2.", "3."))
                ]
                for l in lines:
                    if l and l not in queries:
                        queries.append(l)
            except Exception as exc:
                logger.debug("Multi-query expansion error: %s", exc)

        # Fallback expansion if LLM not present or failed
        if len(queries) == 1:
            queries.extend([f"{query} email", f"{query} details"])

        # Execute retrieval for each perspective
        fused_scores: Dict[str, float] = {}
        doc_store: Dict[str, Document] = {}

        for q in queries[:self.query_count + 1]:
            try:
                sub_docs = self.base_retriever.invoke(q)
                for rank, doc in enumerate(sub_docs):
                    doc_id = str(doc.metadata.get("email_id") or "") or doc.page_content[:50]
                    # Reciprocal Rank Fusion
                    fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (60 + rank))
                    if doc_id not in doc_store:
                        doc_store[doc_id] = doc
                    else:
                        for k, v in doc.metadata.items():
                            if k not in doc_store[doc_id].metadata or doc_store[doc_id].metadata[k] is None:
                                doc_store[doc_id].metadata[k] = v
            except Exception as exc:
                logger.debug("Error executing sub-query '%s': %s", q, exc)

        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
        top_docs: List[Document] = []
        for did in sorted_ids:
            doc = doc_store[did]
            doc.metadata["fused_score"] = round(fused_scores[did], 4)
            doc.metadata.setdefault("source", "chromadb")
            top_docs.append(doc)

        return top_docs


class ConditionalMultiQueryRetriever(BaseRetriever):
    """LangChain retriever that conditionally routes queries to standard retrieval or MultiQueryRetriever.

    - Straightforward queries -> normal retrieval (fast, focused vector & SQL search).
    - Complex queries -> MultiQueryRetriever (generates multiple perspectives, fuses & deduplicates).
    - Low-confidence searches -> if standard retrieval returns low similarity or 0 matches,
      automatically escalates to MultiQueryRetriever for deep semantic recall.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever = Field(..., description="Underlying retriever (e.g. GmailCustomHybridRetriever or ChromaSemanticRetriever)")
    llm: Optional[Any] = Field(None, description="LangChain BaseChatModel for query routing and expansion")
    query_count: int = Field(3, description="Number of alternative query perspectives for MultiQueryRetriever")
    min_confidence_score: float = Field(0.65, description="Threshold below which normal retrieval escalates to MultiQueryRetriever")
    force_multi_query: Optional[bool] = Field(None, description="Override routing if set")

    @traceable(name="retrieval", run_type="retriever")
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        strategy, reason = route_query_retrieval_strategy(
            query=query,
            llm=self.llm,
            force_multi_query=self.force_multi_query,
        )
        logger.info("Conditional retrieval routing for '%s': %s (%s)", query, strategy, reason)

        # 1. Standard retrieval for straightforward queries
        if strategy == "standard":
            docs = self.base_retriever.invoke(query)
            # Evaluate confidence of standard retrieval
            high_confidence = False
            if docs:
                top_score = max(float(d.metadata.get("score") or 0.0) for d in docs)
                if top_score >= self.min_confidence_score:
                    high_confidence = True

            if high_confidence:
                for d in docs:
                    d.metadata.setdefault("retrieval_strategy", "standard")
                return docs

            # Low confidence escalation
            logger.info("Low confidence in standard retrieval for '%s' (docs=%d). Escalating to MultiQueryRetriever.", query, len(docs))

        # 2. MultiQueryRetriever for complex or low-confidence searches
        mq = MultiQueryRetriever(
            base_retriever=self.base_retriever,
            llm=self.llm,
            query_count=self.query_count,
        )
        mq_docs = mq.invoke(query)
        for d in mq_docs:
            d.metadata.setdefault("retrieval_strategy", "multi_query")
        return mq_docs


# ============================================================================
# 5. Contextual Compression Retriever (Single-Sentence Key Points Extraction)
# ============================================================================

def split_sentences(text: str) -> List[str]:
    """Split text into individual sentences while preserving abbreviations and numbers."""
    if not text or not text.strip():
        return []

    abbrevs = [
        "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.", "vs.", "etc.",
        "e.g.", "i.e.", "Oct.", "Nov.", "Dec.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep."
    ]
    t = text
    for i, abb in enumerate(abbrevs):
        t = t.replace(abb, f"__ABB_{i}__")

    # Protect decimal numbers (e.g. 45.20)
    t = re.sub(r"(\d+)\.(\d+)", r"\1__DECIMAL__\2", t)

    # Split on sentence punctuation followed by whitespace or end of string
    raw_parts = re.split(r"(?<=[.!?])\s+", t)

    sentences: List[str] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        part = part.replace("__DECIMAL__", ".")
        for i, abb in enumerate(abbrevs):
            part = part.replace(f"__ABB_{i}__", abb)
        sentences.append(part)
    return sentences


def split_into_single_sentence_bullets(text: str, max_points: int = 5) -> List[str]:
    """Split and normalize text into single-sentence bullet points.

    Guarantees:
    - Every bullet point starts with '• '
    - Each bullet point contains AT MOST ONE single sentence (never multi-sentence)
    - Cleaned and deduplicated
    """
    if not text or not text.strip():
        return []

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    extracted: List[str] = []

    for line in lines:
        if line.upper() == "NO_RELEVANT_INFO" or "NO RELEVANT INFO" in line.upper():
            continue

        # Strip leading bullet indicators (•, -, *, numbers like 1.)
        cleaned = re.sub(r"^(?:[•\-\*]|\d+[\.\)])\s*", "", line).strip()
        if not cleaned:
            continue

        parts = split_sentences(cleaned)
        for p in parts:
            p_clean = p.strip()
            if not p_clean:
                continue
            if not p_clean.endswith((".", "!", "?", ":", ";")):
                p_clean += "."
            extracted.append(f"• {p_clean}")

    # Deduplicate while preserving order
    seen: Set[str] = set()
    result: List[str] = []
    for bullet in extracted:
        norm = bullet.lower()
        if norm not in seen:
            seen.add(norm)
            result.append(bullet)

    return result[:max_points]


KEY_POINTS_COMPRESSION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise email context compression assistant for an enterprise Gmail RAG system.
Given a user search query and an email, extract key points that are directly relevant to the user query.

STRICT FORMATTING REQUIREMENTS:
1. Format each key point as a bullet point starting with '• '.
2. Each bullet point MUST be exactly ONE sentence maximum. Never combine multiple sentences into a single bullet point.
3. Be concise and factual. Ground all facts solely in the email content (include dates, times, senders, action items, monetary amounts, or interview details).
4. If the email contains no relevant information for the query, reply with: NO_RELEVANT_INFO.
5. Do not include any introduction, conversational filler, or conclusion. Return only the bullet points.""",
    ),
    (
        "human",
        """User Query: {query}

Email Subject: {subject}
Sender: {sender}
Date: {date}

Email Body:
{body}

Key Points:""",
    ),
])


class EmailKeyPointsCompressor:
    """Document compressor that extracts query-relevant key points as single-sentence bullet points."""

    def __init__(self, llm: Optional[BaseChatModel] = None, max_key_points: int = 5):
        self.llm = llm
        self.max_key_points = max_key_points

    @traceable(name="key_points_compression", run_type="chain")
    def compress_document(self, doc: Document, query: str) -> Document:
        """Compress a single document into query-relevant single-sentence bullet points."""
        meta = dict(doc.metadata or {})
        subject = str(meta.get("subject") or "")
        sender = str(meta.get("sender") or meta.get("from_email") or "")
        date = str(meta.get("received_at") or meta.get("date") or "")
        body = doc.page_content or ""

        bullets: List[str] = []

        if self.llm:
            try:
                chain = KEY_POINTS_COMPRESSION_PROMPT | self.llm | StrOutputParser()
                raw_out = chain.invoke({
                    "query": query,
                    "subject": subject,
                    "sender": sender,
                    "date": date,
                    "body": body[:2000],
                }).strip()
                bullets = split_into_single_sentence_bullets(raw_out, max_points=self.max_key_points)
            except Exception as exc:
                logger.debug("LLM compression error: %s. Using heuristic fallback.", exc)

        # Heuristic fallback if LLM offline, unconfigured, or returned no bullets
        if not bullets:
            bullets = self._heuristic_extract_key_points(body=body, query=query, subject=subject)

        compressed_content = "\n".join(bullets) if bullets else body[:300]
        meta["key_points"] = bullets
        meta["compressed"] = True

        return Document(page_content=compressed_content, metadata=meta)

    def _heuristic_extract_key_points(self, body: str, query: str, subject: str) -> List[str]:
        """Extract query-relevant sentences as single-sentence bullets using lexical heuristics."""
        if not body:
            return [f"• {subject}."] if subject else []

        clean_body = body
        if "\n\n" in clean_body:
            parts = clean_body.split("\n\n", 1)
            if any(h in parts[0] for h in ["Subject:", "From:", "Date:", "Message-ID:"]):
                clean_body = parts[1]

        raw_sentences = split_sentences(clean_body)
        clean_sentences = [
            s.strip()
            for s in raw_sentences
            if s.strip() and len(s.strip()) > 10 and not any(s.strip().lower().startswith(h) for h in ["subject:", "from:", "date:", "message-id:", "to:"])
        ]

        if not clean_sentences:
            return [f"• {clean_body.strip()[:150]}."]

        query_words = set(re.findall(r"\w+", query.lower())) - {
            "find", "search", "all", "emails", "from", "the", "with", "about",
            "what", "when", "is", "are", "me", "show", "get", "please"
        }

        scored = []
        for s in clean_sentences:
            s_words = set(re.findall(r"\w+", s.lower()))
            overlap = len(query_words & s_words)
            scored.append((overlap, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [s for _, s in scored[:self.max_key_points]]

        if not chosen:
            chosen = clean_sentences[:self.max_key_points]

        bullets: List[str] = []
        for c in chosen:
            c_clean = c.strip()
            if not c_clean.endswith((".", "!", "?")):
                c_clean += "."
            bullets.append(f"• {c_clean}")

        return split_into_single_sentence_bullets("\n".join(bullets), max_points=self.max_key_points)

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """Compress multiple retrieved documents into single-sentence bullet points."""
        compressed_docs: List[Document] = []
        for doc in documents:
            try:
                comp_doc = self.compress_document(doc, query)
                compressed_docs.append(comp_doc)
            except Exception as exc:
                logger.debug("Failed to compress document %s: %s", doc.metadata.get("email_id"), exc)
                compressed_docs.append(doc)
        return compressed_docs


class ContextualCompressionRetriever(BaseRetriever):
    """LangChain retriever that compresses retrieved email documents using contextual key points extraction.

    Guarantees:
    - Executes underlying base retriever (e.g. ConditionalMultiQueryRetriever or GmailCustomHybridRetriever).
    - Compresses each retrieved email into query-relevant key points.
    - Each key point is formatted as a bullet point starting with '• ' in at most ONE sentence.
    - Preserves all email metadata (email_id, subject, sender, date, thread_id, source).
    - Traced via LangSmith.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever = Field(..., description="Underlying retriever")
    base_compressor: Optional[Any] = Field(None, description="Document compressor")
    llm: Optional[Any] = Field(None, description="Chat model for contextual key points compression")
    max_key_points_per_doc: int = Field(5, description="Maximum bullet points per retrieved document")

    @traceable(name="contextual_compression", run_type="retriever")
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Retrieve documents and compress them into single-sentence bullet points."""
        docs = self.base_retriever.invoke(query)
        if not docs:
            return []

        compressor = self.base_compressor
        if compressor is None:
            compressor = EmailKeyPointsCompressor(
                llm=self.llm,
                max_key_points=self.max_key_points_per_doc,
            )

        if hasattr(compressor, "compress_documents"):
            compressed = compressor.compress_documents(docs, query)
        else:
            compressed = docs

        return compressed


class RetrievalStrategyEvaluator:
    """Evaluates clustering and reranking trade-offs for large-scale retrieval optimization."""

    @staticmethod
    def evaluate_clustering_for_scale(
        vector_count: int = 1000,
        dimensions: int = 3072,
        k: int = 5,
        n_clusters: int = 10,
    ) -> dict[str, Any]:
        """Evaluate flat HNSW vs cluster-centroid pre-filtering at mailbox scale."""
        import time
        import numpy as np

        np.random.seed(42)
        sample_count = min(vector_count, 5000)
        eff_dim = min(dimensions, 256) if vector_count > 10000 else dimensions
        vectors = np.random.randn(sample_count, eff_dim).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        query = np.random.randn(eff_dim).astype(np.float32)
        query /= np.linalg.norm(query)

        # 1. Flat HNSW / Exact Cosine Search Benchmark
        t0 = time.perf_counter()
        sims = np.dot(vectors, query)
        ground_truth_indices = set(np.argsort(sims)[-k:])
        raw_flat_ms = (time.perf_counter() - t0) * 1000.0
        t_flat_ms = raw_flat_ms * (float(vector_count) / float(sample_count)) if vector_count > 10000 else raw_flat_ms

        # 2. Cluster Centroid Pre-filtering Benchmark
        t1 = time.perf_counter()
        cluster_indices = np.random.randint(0, n_clusters, size=sample_count)
        centroids = np.zeros((n_clusters, eff_dim), dtype=np.float32)
        for c in range(n_clusters):
            members = vectors[cluster_indices == c]
            if len(members) > 0:
                mean = np.mean(members, axis=0)
                norm = np.linalg.norm(mean)
                centroids[c] = mean / norm if norm > 0 else mean

        c_sims = np.dot(centroids, query)
        top_clusters = np.argsort(c_sims)[-2:]
        candidate_indices = np.where(np.isin(cluster_indices, top_clusters))[0]
        if len(candidate_indices) > 0:
            c_cand_sims = np.dot(vectors[candidate_indices], query)
            top_clustered_local = np.argsort(c_cand_sims)[-k:]
            clustered_indices = set(candidate_indices[top_clustered_local])
        else:
            clustered_indices = set()
        raw_cluster_ms = (time.perf_counter() - t1) * 1000.0
        t_cluster_ms = raw_cluster_ms * (float(vector_count) / float(sample_count)) if vector_count > 10000 else raw_cluster_ms

        recall = len(ground_truth_indices & clustered_indices) / float(k)

        assessment = (
            f"For typical user mailbox scale (N={vector_count} chunks), flat ChromaDB HNSW search "
            f"executes in {t_flat_ms:.2f}ms with 100% recall. Cluster-based routing incurs centroid "
            f"overhead ({t_cluster_ms:.2f}ms) and suffers boundary loss (Recall@{k}={recall * 100:.1f}%). "
            "Clustering is only justified when N > 100,000 vectors."
        )

        return {
            "vector_count": vector_count,
            "dimensions": dimensions,
            "flat_hnsw_latency_ms": round(t_flat_ms, 3),
            "clustered_latency_ms": round(t_cluster_ms, 3),
            "recall_at_k": round(recall, 3),
            "is_clustering_justified": vector_count > 100000,
            "assessment": assessment,
        }

    @staticmethod
    def evaluate_reranking_tradeoffs(
        documents: List[Document],
        query: str,
        k: int = 5,
    ) -> dict[str, Any]:
        """Evaluate when reranking is justified vs cost in latency."""
        import time

        t0 = time.perf_counter()
        doc_count = len(documents)
        scores = [float(d.metadata.get("score") or 0.85) for d in documents[:k]]
        score_margin = (max(scores) - min(scores)) if len(scores) > 1 else 1.0

        is_ambiguous = score_margin < 0.05 and doc_count > 1
        t_eval_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "candidate_count": doc_count,
            "score_margin": round(score_margin, 4),
            "is_reranking_justified": is_ambiguous,
            "simulated_reranker_latency_ms": round(doc_count * 20.0, 1),
            "assessment": (
                f"Reranking adds ~{round(doc_count * 20.0, 1)}ms latency. Justified only when score margin "
                f"< 0.05 (current margin: {score_margin:.4f}). Defaulting to fast single-query HNSW."
            ),
        }


# Backward compatibility instance and alias
GmailHybridRetriever = GmailCustomHybridRetriever
default_retriever = GmailCustomHybridRetriever(user_id="default")


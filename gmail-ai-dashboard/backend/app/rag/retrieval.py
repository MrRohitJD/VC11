"""LangChain retrievers implementation for Gmail RAG system.

Provides 4 specialized retrievers adhering to LangChain's BaseRetriever abstraction:
1. ChromaSemanticRetriever: Dense semantic vector search via ChromaDB (Azure OpenAI text-embedding-3-large)
2. GmailCustomHybridRetriever: Fuses semantic search with PostgreSQL filters (sender, date range, labels, read status, email ID, thread ID)
3. HistoryAwareRetriever: Rewrites conversational follow-up questions into standalone queries using chat history
4. MultiQueryRetriever: Generates alternative search queries, executes parallel retrieval, merges and deduplicates via RRF
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
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
        if "\n\n" in snippet:
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
                snippet=snippet[:300].strip(),
                score=round(float(score), 4),
                thread_id=str(meta.get("thread_id") or ""),
                label=str(meta.get("labels") or meta.get("label") or "EMAIL"),
            )
        )
    return citations


# ============================================================================
# 1. ChromaDB Semantic Retriever
# ============================================================================

class ChromaSemanticRetriever(BaseRetriever):
    """LangChain retriever executing dense semantic vector similarity search via ChromaDB.

    Enforces authenticated user isolation on every query path.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str = Field(..., description="Authenticated user UUID string for tenant isolation")
    top_k: int = Field(5, description="Number of top similar chunks to return")
    filter_dict: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters for ChromaDB")
    score_threshold: Optional[float] = Field(None, description="Minimum similarity threshold")
    chroma_manager: Any = Field(default_factory=lambda: default_chroma_manager)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Retrieve relevant email chunks using ChromaDB dense semantic embeddings."""
        if not query or not query.strip():
            return []

        results = self.chroma_manager.search_with_scores(
            query=query.strip(),
            user_id=self.user_id,
            top_k=self.top_k,
            filter_dict=self.filter_dict,
        )

        docs: List[Document] = []
        for doc, score in results:
            if self.score_threshold is not None and score is not None:
                if score < self.score_threshold:
                    continue

            meta = dict(doc.metadata or {})
            meta["score"] = round(float(score) if score is not None else 0.85, 4)
            docs.append(Document(page_content=doc.page_content, metadata=meta))

        return docs[:self.top_k]

    def retrieve_citations(self, query: str) -> List[RetrievedCitation]:
        """Convenience method returning structured RetrievedCitation objects."""
        return documents_to_citations(self.invoke(query))


# ============================================================================
# 2. Custom Hybrid Retriever (ChromaDB + PostgreSQL Filters)
# ============================================================================

class GmailCustomHybridRetriever(BaseRetriever):
    """Hybrid retriever combining ChromaDB semantic search with PostgreSQL metadata filters.

    Supports structured filtering by:
    - sender (from_email)
    - date range (start_date, end_date)
    - labels (Gmail labels and AI categories)
    - read status (is_read)
    - email_id (exact email UUID)
    - thread_id (thread correspondence)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str = Field(..., description="Authenticated user UUID string")
    db_session: Optional[Any] = Field(None, description="SQLAlchemy Session for relational queries")
    top_k: int = Field(5, description="Total ranked documents to return")

    # Structured metadata filters
    sender: Optional[str] = Field(None, description="Filter by sender address or domain")
    start_date: Optional[datetime] = Field(None, description="Filter emails received on or after")
    end_date: Optional[datetime] = Field(None, description="Filter emails received before")
    label: Optional[str] = Field(None, description="Filter by label or AI category")
    is_read: Optional[bool] = Field(None, description="Filter by read/unread status")
    email_id: Optional[str] = Field(None, description="Filter by specific email UUID")
    thread_id: Optional[str] = Field(None, description="Filter by thread ID")

    semantic_weight: float = Field(0.6, description="Weight for semantic vector scores")
    keyword_weight: float = Field(0.4, description="Weight for structured SQL matches")
    chroma_manager: Any = Field(default_factory=lambda: default_chroma_manager)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Execute hybrid search blending ChromaDB semantic vectors and PostgreSQL metadata."""
        user_uuid = UUID(str(self.user_id))
        semantic_docs: List[Document] = []
        sql_docs: List[Document] = []

        # 1. Semantic search via ChromaDB
        chroma_filter: Dict[str, Any] = {}
        if self.email_id:
            chroma_filter["email_id"] = str(self.email_id)
        if self.thread_id:
            chroma_filter["thread_id"] = str(self.thread_id)

        try:
            raw_semantic = self.chroma_manager.search_with_scores(
                query=query,
                user_id=self.user_id,
                top_k=self.top_k * 2,
                filter_dict=chroma_filter if chroma_filter else None,
            )
            for doc, score in raw_semantic:
                meta = dict(doc.metadata or {})
                meta["score"] = round(float(score) if score is not None else 0.85, 4)
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
                if self.thread_id:
                    db_query = db_query.where(Email.thread_id == self.thread_id.strip())
                if self.sender:
                    db_query = db_query.where(Email.from_email.ilike(f"%{self.sender.strip()}%"))
                if self.start_date:
                    db_query = db_query.where(Email.received_at >= self.start_date)
                if self.end_date:
                    db_query = db_query.where(Email.received_at < self.end_date)
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
                keywords = [w for w in query.split() if len(w) > 3 and w.isalnum()]
                if keywords and not (self.email_id or self.thread_id):
                    search_pat = f"%{' '.join(keywords[:2])}%"
                    db_query = db_query.where(
                        or_(
                            Email.subject.ilike(search_pat),
                            Email.snippet.ilike(search_pat),
                            Email.body.ilike(search_pat),
                        )
                    )

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
                                "subject": e.subject or "No Subject",
                                "sender": e.from_email or "Unknown",
                                "received_at": e.received_at.isoformat() if e.received_at else "",
                                "thread_id": e.thread_id or "",
                                "is_read": e.is_read,
                                "labels": self.label or "EMAIL",
                                "score": 1.0,
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
        limit: int = 5,
    ) -> List[RetrievedCitation]:
        retriever = GmailCustomHybridRetriever(
            user_id=str(user_id),
            db_session=db,
            top_k=limit,
            sender=sender,
            start_date=start_date,
            end_date=end_date,
            label=label,
            is_read=is_read,
            email_id=email_id,
            thread_id=thread_id,
            chroma_manager=self.chroma_manager,
        )
        docs = retriever.invoke(keyword or "")
        return documents_to_citations(docs)

    def retrieve_sql_metadata(
        self,
        db: Session,
        query: str,
        user_id: UUID | str,
        top_k: int = 5,
    ) -> List[RetrievedCitation]:
        return self.retrieve_structured_sql(db=db, user_id=user_id, keyword=query, limit=top_k)

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
# 3. History-Aware Retriever
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
        standalone_query = query

        if self.chat_history:
            # Format chat history text
            history_lines = []
            for msg in self.chat_history[-6:]:
                role = type(msg).__name__.replace("Message", "")
                content = str(msg.content)[:200]
                history_lines.append(f"{role}: {content}")
            history_text = "\n".join(history_lines)

            if self.llm:
                try:
                    chain = self.prompt | self.llm | StrOutputParser()
                    rephrased = chain.invoke({"chat_history": history_text, "input": query}).strip()
                    if rephrased:
                        standalone_query = rephrased
                except Exception as exc:
                    logger.debug("History-aware query rewrite error (%s), using fallback heuristic.", exc)

            # Heuristic fallback if LLM not present or query was unchanged
            if standalone_query == query:
                ambiguous_tokens = {"it", "its", "this", "that", "these", "those", "he", "him", "she", "her", "they", "them"}
                query_words = set(query.lower().replace("?", " ").replace(".", " ").split())
                if any(token in query_words for token in ambiguous_tokens):
                    stop_words = {
                        "did", "i", "receive", "an", "email", "emails", "about", "my", "me", "you", "your",
                        "yes", "no", "a", "an", "the", "for", "at", "on", "in", "to", "from", "with", "is",
                        "was", "are", "were", "what", "when", "where", "how", "why", "who", "have", "has",
                        "had", "do", "does", "can", "could", "would", "should", "will", "please", "tell"
                    }
                    salient: List[str] = []
                    human_msgs = [m for m in self.chat_history if "Human" in type(m).__name__]
                    sources = human_msgs[-2:] if human_msgs else self.chat_history[-2:]
                    for msg in reversed(sources):
                        content = str(msg.content)
                        for w in content.split():
                            clean = w.strip("?,.!:;\"'")
                            if len(clean) > 3 and clean.lower() not in stop_words and clean.lower() not in query_words:
                                if clean.lower() not in [s.lower() for s in salient]:
                                    salient.append(clean)
                    if salient:
                        standalone_query = f"{query} {' '.join(salient[:3])}"

        docs = self.base_retriever.invoke(standalone_query)
        for d in docs:
            d.metadata["rewritten_query"] = standalone_query
        return docs


# ============================================================================
# 4. Multi-Query Retriever
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


class MultiQueryRetriever(BaseRetriever):
    """LangChain retriever generating multiple queries from diverse perspectives, merging and deduplicating results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever = Field(..., description="Underlying retriever (semantic or hybrid)")
    llm: Optional[Any] = Field(None, description="LangChain BaseChatModel for query expansion")
    query_count: int = Field(3, description="Number of alternative query perspectives to generate")
    prompt: ChatPromptTemplate = Field(default=MULTI_QUERY_PROMPT, description="Expansion prompt template")

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
            except Exception as exc:
                logger.debug("Error executing sub-query '%s': %s", q, exc)

        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
        top_docs: List[Document] = []
        for did in sorted_ids:
            doc = doc_store[did]
            doc.metadata["fused_score"] = round(fused_scores[did], 4)
            top_docs.append(doc)

        return top_docs


# Backward compatibility instance and alias
GmailHybridRetriever = GmailCustomHybridRetriever
default_retriever = GmailCustomHybridRetriever(user_id="default")

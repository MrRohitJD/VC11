"""LangChain RAG chain and runnable composition.

Builds grounded email question-answering pipeline using:
- LangChain ChatPromptTemplate
- AzureChatOpenAI (gpt-4.1-mini) / configured LLM provider fallback
- LangChain StrOutputParser
- Explicit citation grounding and hallucination prevention
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from sqlalchemy.orm import Session


try:
    from langsmith import traceable
except ImportError:
    def traceable(name: Optional[str] = None, run_type: Optional[str] = None, **kwargs):
        def decorator(func):
            return func
        return decorator if (name is None or isinstance(name, str)) else name


from app.core.config import settings
from app.rag.retrieval import RetrievedCitation, default_retriever

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are an intelligent Gmail AI Assistant.
Your task is to answer the user's questions about their emails accurately and truthfully.

STRICT INSTRUCTIONS:
1. Base your answer ONLY on the provided email context.
2. DO NOT fabricate, hallucinate, or assume email subjects, senders, dates, or contents that are not in the context.
3. If the provided context is insufficient or contains no matching emails, clearly state: "I could not find relevant emails in your mailbox matching this request."
4. When referencing emails, cite the Sender, Subject, and Date clearly.
5. Provide actionable insights, structured timelines, or next steps when answering about interviews, bills, or tasks.
6. Present the key points extracted from the retrieved emails as bullet points, where each key point is in ONE SENTENCE MAXIMUM.

EMAIL CONTEXT:
{context}
"""

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{question}"),
])


def get_chat_model() -> BaseChatModel:
    """Instantiate and return the configured LangChain chat model.

    Priority:
    1. Azure OpenAI (gpt-4.1-mini) if credentials and endpoint are present.
    2. Fallback to existing LLM provider or standard ChatOpenAI.
    """
    has_azure_key = bool(settings.azure_openai_api_key and settings.azure_openai_api_key.strip())
    has_azure_endpoint = bool(settings.azure_openai_endpoint and settings.azure_openai_endpoint.strip())

    if has_azure_key and has_azure_endpoint:
        try:
            return AzureChatOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                azure_deployment="gpt-4.1-mini",
                temperature=0.1,
            )
        except Exception as exc:
            logger.warning("Could not initialize AzureChatOpenAI (%s). Falling back.", exc)

    if settings.openai_api_key:
        try:
            return ChatOpenAI(
                api_key=settings.openai_api_key,
                model="gpt-4o-mini",
                temperature=0.1,
            )
        except Exception:
            pass

    # Use existing app LLM provider fallback
    try:
        from app.services.llm_provider import get_chat_model as get_app_chat_model
        return get_app_chat_model()
    except Exception:
        # Fallback to local ChatOpenAI with dummy key for offline tests
        return ChatOpenAI(api_key="sk-dummy", model="gpt-4o-mini", temperature=0.1)


def format_context_from_citations(citations: List[RetrievedCitation]) -> str:
    """Format retrieved citations into clean context for LLM prompt."""
    if not citations:
        return "No relevant emails found."

    chunks = []
    for idx, c in enumerate(citations, 1):
        content_lines = "\n".join(c.key_points) if c.key_points else c.snippet
        content_header = "Key Points:" if c.key_points else "Snippet / Content:"
        chunks.append(
            f"--- Email [{idx}] ---\n"
            f"ID: {c.email_id}\n"
            f"Subject: {c.subject}\n"
            f"From: {c.sender}\n"
            f"Date: {c.date}\n"
            f"{content_header}\n{content_lines}\n"
        )
    return "\n".join(chunks)


from app.rag.retrieval import (
    ChromaSemanticRetriever,
    ConditionalMultiQueryRetriever,
    ContextualCompressionRetriever,
    EmailKeyPointsCompressor,
    GmailCustomHybridRetriever,
    HistoryAwareRetriever,
    MultiQueryRetriever,
    RetrievedCitation,
    default_retriever,
    detect_exact_identifier_or_term,
    documents_to_citations,
    rewrite_query_with_history,
    route_query_retrieval_strategy,
    should_fallback_to_hybrid,
    split_into_single_sentence_bullets,
)


@traceable(name="answer_generation", run_type="chain")
def generate_grounded_answer(chain: Any, context_str: str, question: str) -> str:
    """Invoke the LangChain RAG runnable chain to synthesize a grounded answer."""
    return str(chain.invoke({"context": context_str, "question": question}))


def format_grounded_key_points_summary(citations: List[RetrievedCitation], query: str) -> str:
    """Format retrieved citations directly into clean single-sentence key bullet points.

    Guarantees:
    - Every key point is a bullet point.
    - Each bullet point is in one sentence maximum (never multi-sentence).
    - Includes Subject, Sender, and Date when available.
    - Strips email header noise and deduplicates subjects.
    """
    if not citations:
        return "I could not find relevant emails in your mailbox matching this request."

    bullets = []
    seen_subjects = set()

    for c in citations:
        subject = (c.subject or "No Subject").strip()
        sender = (c.sender or "Unknown").strip()
        date_str = str(c.date or "").strip()
        if date_str and " " in date_str:
            date_display = date_str.split(" ")[0]
        elif date_str and "T" in date_str:
            date_display = date_str.split("T")[0]
        else:
            date_display = date_str[:10] if date_str else ""

        subj_key = (subject.lower(), date_display)
        if subj_key in seen_subjects:
            continue
        seen_subjects.add(subj_key)

        meta_parts = []
        if sender and sender != "Unknown":
            clean_sender = sender
            if "<" in sender:
                clean_sender = sender.split("<")[0].strip() or sender.split("<")[1].rstrip(">").strip()
            meta_parts.append(clean_sender)
        if date_display:
            meta_parts.append(date_display)
        meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""

        raw_candidates = []
        for pt in (c.key_points or []):
            pt_clean = pt.strip().lstrip("•-").strip()
            if not any(pt_clean.lower().startswith(h) for h in ["subject:", "from:", "date:", "message-id:", "to:"]):
                raw_candidates.append(pt_clean)

        if not raw_candidates and c.snippet:
            snippet_body = c.snippet
            if "\n\n" in snippet_body:
                snippet_body = snippet_body.split("\n\n", 1)[1]
            for pt in split_into_single_sentence_bullets(snippet_body, max_points=3):
                pt_clean = pt.strip().lstrip("•-").strip()
                if not any(pt_clean.lower().startswith(h) for h in ["subject:", "from:", "date:", "message-id:", "to:"]):
                    raw_candidates.append(pt_clean)

        # Rank candidates by relevance to query keywords and strip invisible unicode
        query_terms = set(re.findall(r"\w+", query.lower())) - {
            "find", "search", "all", "emails", "from", "the", "with", "about",
            "what", "when", "is", "are", "me", "show", "get", "please", "last", "months", "month"
        }
        scored_candidates = []
        for cand in raw_candidates:
            cand_clean = re.sub(r"[\t\r\n\xa0\u200b\u200c\u200d\u200e\u200f\ufeff\u034f\u00ad]+", " ", cand).strip()
            # Collapse multiple spaces
            cand_clean = re.sub(r"\s+", " ", cand_clean).strip()
            if re.search(r"[a-zA-Z]{3,}", cand_clean) and len(cand_clean) > 15:
                cand_words = set(re.findall(r"\w+", cand_clean.lower()))
                overlap = len(query_terms & cand_words)
                scored_candidates.append((overlap, cand_clean))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        chosen_sentence = ""
        if scored_candidates:
            top_cand = scored_candidates[0][1]
            match = re.search(r"^(.*?[.!?])(?:\s+|$)", top_cand)
            first_sent = match.group(1).strip() if match else top_cand
            if not first_sent.endswith((".", "!", "?")):
                first_sent += "."
            chosen_sentence = first_sent

        if not chosen_sentence:
            chosen_sentence = f"Relevant correspondence regarding '{subject}' recorded in mailbox."

        bullets.append(f"• **{subject}**{meta_str}: {chosen_sentence}")

    clean_q = query.strip()
    return f"Here are the key points from your relevant emails for '{clean_q}':\n\n" + "\n".join(bullets)



class GroundedRAGChain:
    """LangChain runnable pipeline for grounded email question answering with optimized retriever composition.

    Retrieval Strategy:
    1. Default: LangChain + ChromaDB HNSW cosine similarity search with a single query embedding (k=5..10).
    2. Tenant Isolation: user_id is strictly validated and enforced on every retrieval path.
    3. Metadata Filtering: sender, gmail_message_id, received_date, received_timestamp applied directly in Chroma.
    4. Optional Fallback: Hybrid keyword (PostgreSQL) + vector (ChromaDB) retrieval with Reciprocal Rank Fusion (RRF)
       for exact terms, names, IDs, and low-confidence searches.
    5. Optional MMR: search_type="mmr" supported for chunk diversity, disabled by default for low latency.
    6. Conditional MultiQuery: only executed for complex multi-topic queries or explicit escalation.
    """

    def __init__(self, retriever=default_retriever, model: Optional[BaseChatModel] = None):
        self.retriever = retriever
        self.model = model or get_chat_model()
        self.output_parser = StrOutputParser()

        # Build LangChain Runnable composition
        self.chain = RAG_PROMPT_TEMPLATE | self.model | self.output_parser

    @traceable(name="rag_answer_query", run_type="chain")
    def answer_query(
        self,
        db: Session,
        query: str,
        user_id: UUID | str,
        top_k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
        chat_history: Optional[List[BaseMessage]] = None,
        use_multi_query: Optional[bool] = None,
        retrieval_mode: str = "similarity",
    ) -> dict[str, Any]:
        """Execute optimized retrieval and grounded answer generation.

        Args:
            db: SQLAlchemy Session for relational data and hybrid fallback.
            query: User's question or search phrase.
            user_id: Authenticated user UUID string for tenant isolation.
            top_k: Number of documents to retrieve (default 5, configurable 5-10).
            filter_dict: Optional metadata filters (sender, gmail_message_id, start_date, end_date, etc.).
            chat_history: Optional recent conversation turns for query rewriting.
            use_multi_query: Explicit override for multi-query retrieval (None for auto).
            retrieval_mode: Retrieval mode ('similarity' for default HNSW, 'mmr' for diversity, 'hybrid' for SQL+vector).
        """
        filters = dict(filter_dict or {})
        user_str = str(user_id).strip()
        if not user_str:
            raise ValueError("user_id is mandatory for tenant isolation on every retrieval path")

        # 1. Step 1: LLM-based query rewriting using conversation history if present
        effective_query = query
        if chat_history:
            effective_query = rewrite_query_with_history(
                query=query,
                chat_history=chat_history,
                llm=self.model,
            )
            logger.info("Rewrote ambiguous query '%s' -> '%s'", query, effective_query)

        # 2. Step 2: Determine retrieval strategy
        is_hybrid_mode = (retrieval_mode == "hybrid")
        has_exact, exact_reason = detect_exact_identifier_or_term(effective_query, filters)
        search_type = "mmr" if retrieval_mode == "mmr" else "similarity"

        # Primary fast retriever: LangChain + ChromaDB HNSW similarity search
        semantic_retriever = ChromaSemanticRetriever(
            user_id=user_str,
            top_k=top_k,
            search_type=search_type,
            sender=filters.get("sender"),
            gmail_message_id=filters.get("gmail_message_id"),
            start_date=filters.get("start_date"),
            end_date=filters.get("end_date"),
            start_timestamp=filters.get("start_timestamp"),
            end_timestamp=filters.get("end_timestamp"),
            filter_dict=filters.get("custom_filter"),
        )

        retrieval_strategy_used = "standard"
        docs: List[Document] = []

        if not is_hybrid_mode and not has_exact and use_multi_query is not True:
            # Fast default path: single query embedding through ChromaDB HNSW cosine search
            docs = semantic_retriever.invoke(effective_query)
            if docs:
                retrieval_strategy_used = "standard"

        # 3. Step 3: Check if fallback to Hybrid Keyword + Vector Retrieval with RRF is needed
        needs_fallback, fallback_reason = should_fallback_to_hybrid(
            query=effective_query,
            semantic_docs=docs,
            filters=filters,
            min_confidence_score=0.65,
        )

        if is_hybrid_mode or has_exact or needs_fallback or use_multi_query is True:
            logger.info(
                "Using Hybrid Keyword + Vector Retrieval with RRF (exact=%s: %s, fallback=%s: %s)",
                has_exact,
                exact_reason,
                needs_fallback,
                fallback_reason,
            )
            hybrid_retriever = GmailCustomHybridRetriever(
                user_id=user_str,
                db_session=db,
                top_k=top_k,
                sender=filters.get("sender"),
                gmail_message_id=filters.get("gmail_message_id"),
                start_date=filters.get("start_date"),
                end_date=filters.get("end_date"),
                start_timestamp=filters.get("start_timestamp"),
                end_timestamp=filters.get("end_timestamp"),
                label=filters.get("label"),
                is_read=filters.get("is_read"),
                email_id=filters.get("email_id"),
                thread_id=filters.get("thread_id"),
                search_type=search_type,
            )

            # Conditional MultiQueryRetriever: only for complex multi-topic queries or explicit flag
            strategy_cls, _ = route_query_retrieval_strategy(effective_query, llm=None, force_multi_query=use_multi_query)
            if strategy_cls == "multi_query":
                mq_retriever = MultiQueryRetriever(
                    base_retriever=hybrid_retriever,
                    llm=self.model,
                    query_count=3,
                )
                docs = mq_retriever.invoke(effective_query)
                retrieval_strategy_used = "multi_query"
            else:
                docs = hybrid_retriever.invoke(effective_query)
                retrieval_strategy_used = "standard"

        # 4. Step 4: Contextual Compression Retriever extracting single-sentence key points
        compressor = EmailKeyPointsCompressor(llm=self.model, max_key_points=5)
        compressed_docs = compressor.compress_documents(docs, effective_query)
        citations = documents_to_citations(compressed_docs)

        # Fallback to broader hybrid search if zero matches and specific filters were applied
        if not citations and filters and db is not None:
            citations = self.retriever.retrieve_hybrid(
                db=db,
                query=effective_query,
                user_id=user_id,
                top_k=top_k,
            )
            for c in citations:
                if not c.key_points and c.snippet:
                    c.key_points = split_into_single_sentence_bullets(c.snippet, max_points=3)

        # 5. Format context from deduplicated citations with compressed key points
        context_str = format_context_from_citations(citations)

        # 6. Step 5: Generate grounded answer through traced LangChain runnable
        try:
            answer_text = generate_grounded_answer(self.chain, context_str, effective_query)
        except Exception as exc:
            logger.warning("LLM invocation error: %s. Using deterministic grounded summary with single-sentence bullets.", exc)
            answer_text = format_grounded_key_points_summary(citations, effective_query)

        return {
            "answer": answer_text,
            "citations": citations,
            "context_count": len(citations),
            "rewritten_query": effective_query,
            "retrieval_strategy": retrieval_strategy_used,
        }



default_rag_chain = GroundedRAGChain()

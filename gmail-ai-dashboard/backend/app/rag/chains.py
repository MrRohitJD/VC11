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
from functools import lru_cache
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
from app.rag.query_constraints import mailbox_now, is_financial_query
from app.rag.mailbox_scope import primary_email_ids
from app.rag.relevance import validate_email_relevance

logger = logging.getLogger(__name__)

USER_RESPONSE_STYLE = """
Write for the mailbox owner, using plain language and addressing them as "you".
Mailbox searches and answers use only Gmail's Primary category. If no emails
match, explicitly say no matching emails were found in Primary. Do not claim
to have searched Social, Promotions, Updates, Forums, or the entire mailbox.
Focus on their question and what they need to know or do. Do not describe internal
routing, tools, execution plans, RAG, vector databases, or model configuration.
For substantive mailbox answers, use this Markdown structure:
## Summary
Give the direct answer in one or two short sentences.
## Key details
Use concise bullets for relevant people, dates, amounts, and deadlines.
## Next steps
Include only when an action is relevant; use a short numbered list.
## Related emails
List supporting subjects, senders, and dates only when available.
Omit empty sections. For greetings and simple questions, reply briefly without
forcing headings. If nothing matches, say so and suggest a useful search refinement.
Never invent facts, next steps, or sources to fill the structure. Describe drafts
as awaiting review; only say an email was sent when the tool confirms success.
"""

RAG_SYSTEM_PROMPT = """You are an intelligent Gmail AI Assistant.
Your task is to answer the user's questions about their emails accurately and truthfully.

STRICT INSTRUCTIONS:
1. Base your answer ONLY on the provided email context.
2. DO NOT fabricate, hallucinate, or assume email subjects, senders, dates, or contents that are not in the context.
3. If the provided context is insufficient or contains no matching emails, clearly state: "I could not find relevant emails in your Primary category matching this request."
4. When referencing emails, cite the Sender, Subject, and Date clearly.
5. Provide actionable insights, structured timelines, or next steps when answering about interviews, bills, or tasks.
6. Present the key points extracted from the retrieved emails as bullet points, where each key point is in ONE SENTENCE MAXIMUM.

EMAIL CONTEXT:
{context}

For bills or invoices, distinguish unpaid balances from receipts and paid
confirmations. A message received this month is not evidence that its payment
is due this month. Use explicit payment due dates and preserve their year.
If payment status or the due date is unclear, state the uncertainty.
""" + USER_RESPONSE_STYLE

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{question}"),
])


@lru_cache(maxsize=1)
def get_chat_model() -> Any:
    """Instantiate and return the configured LangChain chat model.

    Priority:
    NVIDIA first, followed by configured Azure/OpenAI and ExperientialLabs.
    Failures use real provider fallbacks; no dummy credentials or local verdicts.
    """
    models = []
    if settings.nvidia_api_key:
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            models.append(ChatNVIDIA(
                model="openai/gpt-oss-20b", api_key=settings.nvidia_api_key,
                temperature=0.1, max_completion_tokens=4096, timeout=60,
                model_kwargs={"reasoning_effort": "low"},
            ))
        except Exception as exc:
            logger.warning("NVIDIA assistant provider unavailable: %s", type(exc).__name__)
    has_azure_key = bool(settings.azure_openai_api_key and settings.azure_openai_api_key.strip())
    has_azure_endpoint = bool(settings.azure_openai_endpoint and settings.azure_openai_endpoint.strip())

    if has_azure_key and has_azure_endpoint:
        try:
            models.append(AzureChatOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                azure_deployment=getattr(settings, "azure_openai_deployment", "gpt-4.1-mini"),
                temperature=0.1,
                timeout=20, max_retries=0,
            ))
        except Exception as exc:
            logger.warning("Could not initialize AzureChatOpenAI (%s). Falling back.", exc)

    if settings.openai_api_key:
        try:
            models.append(ChatOpenAI(
                api_key=settings.openai_api_key,
                model="gpt-4o-mini",
                temperature=0.1,
                timeout=20, max_retries=0,
            ))
        except Exception:
            pass

    if settings.explabs_api_key:
        models.append(ChatOpenAI(
            model="deepseek-v4-flash", base_url="https://api.experientiallabs.ai/v1",
            api_key=settings.explabs_api_key, temperature=0.1, timeout=20, max_retries=0,
        ))
    if models:
        return models[0].with_fallbacks(models[1:]) if len(models) > 1 else models[0]
    # No dummy credentials: offline/unconfigured requests fail explicitly.
    from langchain_core.runnables import RunnableLambda
    def unavailable(_input):
        raise RuntimeError("No assistant LLM provider is configured")
    return RunnableLambda(unavailable)


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
    hydrate_email_documents,
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
        if is_financial_query(query):
            return "I could not verify any matching bills or invoices in your Primary category with the requested payment status and due date."
        return "I could not find relevant emails in your Primary category matching this request."

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
        all_requested = bool(re.search(r"\b(?:find|show|list|get|search)\s+all\b", effective_query, re.I))
        candidate_k = max(top_k * 4, 20)
        if all_requested:
            candidate_k = max(1, len(primary_email_ids(db, user_id, effective_query)))
        is_hybrid_mode = (retrieval_mode == "hybrid")
        has_exact, exact_reason = detect_exact_identifier_or_term(effective_query, filters)
        search_type = "mmr" if retrieval_mode == "mmr" else "similarity"

        # Primary fast retriever: LangChain + ChromaDB HNSW similarity search
        semantic_retriever = ChromaSemanticRetriever(
            user_id=user_str,
            db_session=db,
            primary_only=True,
            scope_query=effective_query,
            top_k=candidate_k,
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
                primary_only=True,
                scope_query=effective_query,
                top_k=candidate_k,
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

        # Use current database metadata and full bodies even for legacy indexes.
        docs = hydrate_email_documents(docs, db, user_id, full_body=True)
        eligible_ids = set(primary_email_ids(db, user_id, effective_query))
        docs = [doc for doc in docs if str(doc.metadata.get("email_id")) in eligible_ids]
        unique_docs = {}
        for doc in docs:
            unique_docs.setdefault(str(doc.metadata.get("email_id")), doc)
        docs = list(unique_docs.values())

        # The LLM receives real envelope metadata and content, then explicitly
        # accepts/rejects each candidate before it can become a cited source.
        validation = validate_email_relevance(docs, effective_query, self.model)
        accepted = validation.documents if all_requested else validation.documents[:top_k]
        citations = documents_to_citations(accepted)

        # 5. Format context from deduplicated citations with compressed key points
        context_str = (
            f"Current date: {mailbox_now().date().isoformat()} ({mailbox_now().tzinfo}).\n\n"
            + format_context_from_citations(citations)
        )

        # 6. Step 5: Generate grounded answer through traced LangChain runnable
        if validation.failed:
            answer_text = "I could not validate which emails match your question because the relevance check is unavailable. Please try again."
        elif not citations:
            answer_text = format_grounded_key_points_summary(citations, effective_query)
        else:
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
            "relevance_validated": not validation.failed,
            "candidates_evaluated": validation.evaluated,
        }



default_rag_chain = GroundedRAGChain()

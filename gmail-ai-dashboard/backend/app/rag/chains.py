"""LangChain RAG chain and runnable composition.

Builds grounded email question-answering pipeline using:
- LangChain ChatPromptTemplate
- AzureChatOpenAI (gpt-4.1-mini) / configured LLM provider fallback
- LangChain StrOutputParser
- Explicit citation grounding and hallucination prevention
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from sqlalchemy.orm import Session

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
                timeout=15,
            )
        except Exception as exc:
            logger.warning("Could not initialize AzureChatOpenAI (%s). Falling back.", exc)

    # Priority 2: NVIDIA NIM
    if settings.nvidia_api_key:
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            return ChatNVIDIA(
                model="openai/gpt-oss-20b",
                api_key=settings.nvidia_api_key,
                temperature=0.1,
                timeout=15,
            )
        except Exception as exc:
            logger.debug("ChatNVIDIA initialization skipped (%s)", exc)

    # Priority 3: ExperientialLabs DeepSeek
    if settings.explabs_api_key:
        try:
            return ChatOpenAI(
                model="deepseek-v4-flash",
                base_url="https://api.experientiallabs.ai/v1",
                api_key=settings.explabs_api_key,
                temperature=0.1,
                timeout=15,
            )
        except Exception as exc:
            logger.debug("ExperientialLabs ChatOpenAI initialization skipped (%s)", exc)

    # Priority 4: Standard ChatOpenAI
    if settings.openai_api_key:
        try:
            return ChatOpenAI(
                api_key=settings.openai_api_key,
                model="gpt-4o-mini",
                temperature=0.1,
                timeout=15,
            )
        except Exception:
            pass

    # Fallback to local ChatOpenAI with dummy key for offline tests
    return ChatOpenAI(api_key="sk-dummy", model="gpt-4o-mini", temperature=0.1)


def format_context_from_citations(citations: List[RetrievedCitation]) -> str:
    """Format retrieved citations into clean context for LLM prompt."""
    if not citations:
        return "No relevant emails found."

    chunks = []
    for idx, c in enumerate(citations, 1):
        chunks.append(
            f"--- Email [{idx}] ---\n"
            f"ID: {c.email_id}\n"
            f"Subject: {c.subject}\n"
            f"From: {c.sender}\n"
            f"Date: {c.date}\n"
            f"Snippet / Content:\n{c.snippet}\n"
        )
    return "\n".join(chunks)


from app.rag.retrieval import (
    GmailCustomHybridRetriever,
    HistoryAwareRetriever,
    MultiQueryRetriever,
    RetrievedCitation,
    default_retriever,
    documents_to_citations,
)


class GroundedRAGChain:
    """LangChain runnable pipeline for grounded email question answering with retriever composition."""

    def __init__(self, retriever=default_retriever, model: Optional[BaseChatModel] = None):
        self.retriever = retriever
        self.model = model or get_chat_model()
        self.output_parser = StrOutputParser()

        # Build LangChain Runnable composition
        self.chain = RAG_PROMPT_TEMPLATE | self.model | self.output_parser

    def answer_query(
        self,
        db: Session,
        query: str,
        user_id: UUID | str,
        top_k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
        chat_history: Optional[List[BaseMessage]] = None,
        use_multi_query: bool = False,
    ) -> dict[str, Any]:
        """Execute retrieval and grounded answer generation using composed LangChain retrievers."""
        filters = dict(filter_dict or {})
        user_str = str(user_id)

        # 1. Base Hybrid Retriever with user isolation and PostgreSQL filters
        base_retriever = GmailCustomHybridRetriever(
            user_id=user_str,
            db_session=db,
            top_k=top_k,
            sender=filters.get("sender"),
            start_date=filters.get("start_date"),
            end_date=filters.get("end_date"),
            label=filters.get("label"),
            is_read=filters.get("is_read"),
            email_id=filters.get("email_id"),
            thread_id=filters.get("thread_id"),
        )

        active_retriever = base_retriever

        # 2. History-Aware Retriever composition
        if chat_history:
            active_retriever = HistoryAwareRetriever(
                base_retriever=active_retriever,
                llm=self.model,
                chat_history=chat_history,
            )

        # 3. Multi-Query Retriever composition
        if use_multi_query:
            active_retriever = MultiQueryRetriever(
                base_retriever=active_retriever,
                llm=self.model,
                query_count=3,
            )

        # Retrieve documents
        docs = active_retriever.invoke(query)
        citations = documents_to_citations(docs)

        # If no citations found and filters were applied, fallback to broader hybrid search
        if not citations and (filters or use_multi_query):
            citations = self.retriever.retrieve_hybrid(
                db=db,
                query=query,
                user_id=user_id,
                top_k=top_k,
            )

        # 4. Format context
        context_str = format_context_from_citations(citations)

        # 5. Generate answer through LangChain runnable
        try:
            raw_answer = self.chain.invoke({
                "context": context_str,
                "question": query,
            })
            answer_text = str(raw_answer)
        except Exception as exc:
            logger.warning("LLM invocation error: %s. Using deterministic grounded summary.", exc)
            if citations:
                answer_text = (
                    f"Based on your retrieved emails, I found {len(citations)} relevant message(s) "
                    f"answering '{query}'. Please see the verified source citations below."
                )
            else:
                answer_text = f"I could not locate any emails matching '{query}' in your synchronized mailbox."

        return {
            "answer": answer_text,
            "citations": citations,
            "context_count": len(citations),
        }


default_rag_chain = GroundedRAGChain()

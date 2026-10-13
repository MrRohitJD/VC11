"""LangChain-native Gmail AI Assistant implementing 10 production agent patterns without LangGraph.

Implemented Patterns:
1. RAG Pattern: Grounded question answering with citations
2. Tool-Calling Agent: Direct LLM tool binding with bounded ReAct execution loop
3. Router Pattern: Intent-based routing to optimal pipeline
4. ReAct-Style Behavior: Explicit Thought -> Action -> Observation reasoning loop
5. Plan-and-Execute: Multi-step query decomposition and sequential execution
6. Human-in-the-Loop: Strict approval gating for email drafts and actions
7. Query Transformation: Standalone question rewriting from conversational history
8. Contextual Compression: Relevance filtering and deduplication of retrieved chunks
9. Hybrid Retrieval: Combining dense ChromaDB vector search with SQL metadata filters
10. Fallback Routing: SQL keyword and deterministic extraction fallbacks
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.orm import Session

from app.models.rag import ChatMessage, ChatSession, EmailSendApproval
from app.rag.chains import GroundedRAGChain, get_chat_model
from app.rag.memory import (
    PostgresChatMessageHistory,
    get_or_create_chat_session,
    record_retrieval_citations,
)
from app.rag.retrieval import GmailHybridRetriever, RetrievedCitation, default_retriever
from app.rag.tools import create_gmail_tools

logger = logging.getLogger(__name__)


# ============================================================================
# Prompts for Router, Query Transformation, Planner, and Agent
# ============================================================================

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an intent router for a Gmail AI assistant.
Analyze the user's message and select the single most appropriate intent category:
- RAG_QUERY: Question about specific emails, dates, senders, bills, interviews, or facts stored in the mailbox.
- ACTION_DRAFT: Request to write, draft, or compose a new email or reply.
- ACTION_SEND: Request to send an approved email or authorize sending.
- PLAN_AND_EXECUTE: Complex multi-step instructions (e.g., "Find all interview emails, summarize them, and draft a response to John").
- SUMMARIZE: General request to summarize the inbox, unread mail, or high-priority emails.
- GENERAL_CHAT: Greeting, capability question, or general conversation.

Respond with ONLY the intent name in uppercase (e.g., RAG_QUERY).""",
    ),
    ("human", "{query}"),
])

QUERY_TRANSFORM_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given the following conversation history and the latest user question,
rewrite the question into a clear, self-contained search query for email retrieval.
Resolve all pronouns and conversational references (e.g., "he", "that email", "the meeting").
DO NOT answer the question. Only return the rewritten query. If the question is already self-contained, return it unchanged.""",
    ),
    ("human", "Conversation History:\n{history}\n\nLatest Question: {query}"),
])

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a task planner for a Gmail assistant.
Break down the complex user request into 2 to 4 concise, sequential sub-tasks.
Format your output as a JSON list of strings, for example:
["Search for emails regarding interview schedules", "Extract recruiter contact details", "Draft thank you email"]
Return ONLY valid JSON.""",
    ),
    ("human", "{query}"),
])

AGENT_SYSTEM_PROMPT = """You are an intelligent, secure Gmail AI Assistant with full tool access to the user's mailbox.
You assist users by searching emails, summarizing correspondence, extracting details, and drafting emails.

GUIDELINES:
1. Always ground your facts in the tools' returned data. Never invent email IDs, dates, or senders.
2. For consequential actions like drafting or sending emails, ALWAYS inform the user that their draft is pending review.
3. Keep answers clear, structured, and professional.
4. If a tool returns no results, explain that no matching records were found and suggest alternative searches.
"""


@dataclass
class AgentStep:
    thought: str = ""
    tool_name: str = ""
    tool_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""


@dataclass
class AgentResponse:
    answer: str
    intent: str
    citations: List[RetrievedCitation] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    plan: Optional[List[str]] = None
    approval_required: bool = False
    approval_id: Optional[str] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None


class GmailAIAssistant:
    """Production Gmail assistant orchestrating 10 agent patterns using LangChain without LangGraph."""

    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        retriever: Optional[GmailHybridRetriever] = None,
        max_agent_iterations: int = 6,
    ):
        self.model = model or get_chat_model()
        self.retriever = retriever or default_retriever
        self.rag_chain = GroundedRAGChain(retriever=self.retriever, model=self.model)
        self.max_agent_iterations = max_agent_iterations
        self.output_parser = StrOutputParser()

    # ------------------------------------------------------------------------
    # Pattern 3: Router Pattern
    # ------------------------------------------------------------------------
    def route_intent(self, query: str) -> str:
        """Classify user query into an actionable routing category."""
        clean_q = query.strip().lower()

        # Fast heuristic checks
        if any(w in clean_q for w in ["send the draft", "approve send", "send approval", "send email", "send this draft"]):
            return "ACTION_SEND"
        if (" and " in clean_q or "then" in clean_q) and len(clean_q.split()) > 10:
            if any(w in clean_q for w in ["draft", "summarize", "find", "extract"]):
                return "PLAN_AND_EXECUTE"
        if any(w in clean_q for w in ["summarize my inbox", "summary of emails", "summarize unread", "inbox recap"]):
            return "SUMMARIZE"
        if any(w in clean_q for w in ["draft", "compose", "write an email", "prepare an email"]):
            return "ACTION_DRAFT"

        # LLM router classification
        try:
            chain = ROUTER_PROMPT | self.model | self.output_parser
            decision = chain.invoke({"query": query}).strip()
            valid_intents = {"RAG_QUERY", "ACTION_DRAFT", "ACTION_SEND", "PLAN_AND_EXECUTE", "SUMMARIZE", "GENERAL_CHAT"}
            for vi in valid_intents:
                if vi in decision:
                    return vi
        except Exception as exc:
            logger.debug("Router LLM invocation fallback: %s", exc)

        return "RAG_QUERY"

    # ------------------------------------------------------------------------
    # Pattern 7: Query Transformation
    # ------------------------------------------------------------------------
    def transform_query(self, query: str, history: List[BaseMessage]) -> str:
        """Rewrite contextual follow-up questions into standalone retrieval queries."""
        if not history:
            return query

        # Format past few turns
        history_lines = []
        for msg in history[-4:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = str(msg.content)[:250]
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

        try:
            chain = QUERY_TRANSFORM_PROMPT | self.model | self.output_parser
            transformed = chain.invoke({"history": history_text, "query": query}).strip()
            if transformed and len(transformed) > 3:
                return transformed
        except Exception as exc:
            logger.debug("Query transformation error (%s), using original query.", exc)

        return query

    # ------------------------------------------------------------------------
    # Pattern 8: Contextual Compression
    # ------------------------------------------------------------------------
    def compress_context(
        self,
        citations: List[RetrievedCitation],
        max_chunks: int = 5,
        min_relevance: float = 0.35,
    ) -> List[RetrievedCitation]:
        """Filter out low-relevance citations and deduplicate across chunks."""
        seen_emails = set()
        compressed = []

        for cit in citations:
            if cit.score is not None and cit.score < min_relevance:
                continue

            # Prioritize first high-score chunk per email
            if cit.email_id in seen_emails and len(seen_emails) >= 3:
                continue

            seen_emails.add(cit.email_id)
            compressed.append(cit)
            if len(compressed) >= max_chunks:
                break

        return compressed or citations[:max_chunks]

    # ------------------------------------------------------------------------
    # Pattern 5: Plan-and-Execute Workflow
    # ------------------------------------------------------------------------
    def execute_plan_and_execute(
        self,
        query: str,
        db: Session,
        user_id: UUID | str,
        tools: List[Any],
    ) -> tuple[str, List[str], List[Dict[str, Any]], List[RetrievedCitation]]:
        """Decompose complex multi-step user goal into sequential sub-tasks."""
        # 1. Generate plan
        plan = []
        try:
            chain = PLANNER_PROMPT | self.model | self.output_parser
            raw_plan = chain.invoke({"query": query}).strip()
            # Extract JSON list
            match = re.search(r"\[.*\]", raw_plan, re.DOTALL)
            if match:
                plan = json.loads(match.group(0))
        except Exception as exc:
            logger.debug("Planner JSON error (%s), falling back to rule-based plan.", exc)
            plan = [
                f"Search emails relevant to: {query}",
                "Analyze and extract key information",
                "Formulate comprehensive response or draft",
            ]

        # 2. Sequential execution
        execution_steps = []
        all_citations: List[RetrievedCitation] = []
        accumulated_context = []

        for idx, subtask in enumerate(plan, 1):
            logger.info("Executing plan step %d/%d: %s", idx, len(plan), subtask)
            # Retrieve subtask context
            sub_citations = self.retriever.retrieve_hybrid(db, subtask, user_id, top_k=3)
            all_citations.extend(sub_citations)

            # ReAct tool step for this subtask
            ans, citations, steps = self.execute_react_agent(
                query=f"{subtask}\nPrior Context: {' '.join(accumulated_context[-2:])}",
                db=db,
                user_id=user_id,
                tools=tools,
                max_iterations=3,
            )
            execution_steps.append({
                "subtask": subtask,
                "output": ans,
                "tool_steps": steps,
            })
            accumulated_context.append(f"Step {idx} ({subtask}): {ans}")

        # 3. Final synthesis
        synth_prompt = (
            f"You are the Gmail Assistant. The user asked: '{query}'\n\n"
            f"Here are the executed step results:\n"
            + "\n\n".join(accumulated_context)
            + "\n\nSynthesize these results into a clear, unified final response."
        )
        try:
            final_answer = self.model.invoke([HumanMessage(content=synth_prompt)]).content
        except Exception:
            final_answer = "\n\n".join(accumulated_context)

        return str(final_answer), plan, execution_steps, all_citations

    # ------------------------------------------------------------------------
    # Pattern 2 & 4: Tool-Calling Agent with ReAct-Style Loop
    # ------------------------------------------------------------------------
    def execute_react_agent(
        self,
        query: str,
        db: Session,
        user_id: UUID | str,
        tools: List[Any],
        max_iterations: Optional[int] = None,
    ) -> tuple[str, List[RetrievedCitation], List[Dict[str, Any]]]:
        """Execute bounded tool-calling reasoning loop without LangGraph."""
        iterations_limit = max_iterations or self.max_agent_iterations
        tool_map = {t.name: t for t in tools}

        # Bind tools to chat model
        model_with_tools = self.model.bind_tools(tools)

        messages: List[BaseMessage] = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]

        steps: List[Dict[str, Any]] = []
        citations_collected: List[RetrievedCitation] = []

        for step_idx in range(iterations_limit):
            try:
                response: AIMessage = model_with_tools.invoke(messages)
            except Exception as exc:
                logger.warning("Agent invocation step %d error: %s", step_idx, exc)
                break

            messages.append(response)

            # Check if model called tools
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                # Agent concluded with final answer
                return str(response.content), citations_collected, steps

            # Execute tool calls
            for tcall in tool_calls:
                tname = tcall.get("name")
                targs = tcall.get("args", {})
                tcall_id = tcall.get("id") or str(uuid4())

                tool_fn = tool_map.get(tname)
                if not tool_fn:
                    obs = f"Error: Tool '{tname}' not recognized."
                else:
                    try:
                        obs = str(tool_fn.invoke(targs))
                    except Exception as err:
                        obs = f"Tool '{tname}' execution error: {err}"

                # If semantic search or context retrieval was called, populate citations
                if tname in ["search_emails", "retrieve_email_context"]:
                    try:
                        if isinstance(obs, str) and obs.startswith("["):
                            parsed_cits = json.loads(obs)
                            for pc in parsed_cits:
                                if isinstance(pc, dict) and "email_id" in pc:
                                    citations_collected.append(
                                        RetrievedCitation(
                                            email_id=pc["email_id"],
                                            subject=pc.get("subject", ""),
                                            sender=pc.get("sender", ""),
                                            date=pc.get("date", ""),
                                            snippet=pc.get("snippet", ""),
                                            score=pc.get("relevance_score", 0.8),
                                        )
                                    )
                    except Exception:
                        pass

                steps.append({
                    "step": step_idx + 1,
                    "thought": str(response.content) or f"Decided to call {tname}",
                    "action": tname,
                    "arguments": targs,
                    "observation": obs[:500] if len(obs) > 500 else obs,
                })

                messages.append(
                    ToolMessage(
                        content=obs,
                        tool_call_id=tcall_id,
                    )
                )

        # Fallback if max iterations reached
        final_text = str(messages[-1].content) if isinstance(messages[-1], AIMessage) else (
            "I completed several email search steps to gather details for your request. "
            "Please see the actions and citations above."
        )
        return final_text, citations_collected, steps

    # ------------------------------------------------------------------------
    # Pattern 10: Fallback Routing & Deterministic Handling
    # ------------------------------------------------------------------------
    def execute_with_fallback(
        self,
        query: str,
        db: Session,
        user_id: UUID | str,
        tools: List[Any],
    ) -> tuple[str, List[RetrievedCitation], List[Dict[str, Any]]]:
        """Attempt primary tool-calling loop; fall back to hybrid RAG and SQL search."""
        try:
            ans, citations, steps = self.execute_react_agent(query, db, user_id, tools)
            if ans and len(ans.strip()) > 0:
                return ans, citations, steps
        except Exception as exc:
            logger.warning("Tool-calling agent error (%s). Falling back to GroundedRAGChain.", exc)

        # Fallback 1: Grounded RAG Chain
        try:
            rag_res = self.rag_chain.answer_query(db, query, user_id, top_k=5)
            return rag_res["answer"], rag_res["citations"], [{"step": 1, "thought": "RAG Fallback", "action": "hybrid_retrieval", "observation": f"Retrieved {len(rag_res['citations'])} emails"}]
        except Exception as exc:
            logger.error("RAG chain fallback also failed (%s). Using SQL keyword fallback.", exc)

        # Fallback 2: Direct SQL Search
        citations = self.retriever.retrieve_sql_metadata(db, query, user_id, top_k=3)
        if citations:
            lines = [f"- {c.subject} from {c.sender} ({c.date})" for c in citations]
            msg = f"I retrieved these matching emails from your mailbox:\n" + "\n".join(lines)
            return msg, citations, [{"step": 1, "thought": "Direct SQL Fallback", "action": "sql_search", "observation": "Found matching emails"}]

        return (
            "I could not find matching emails in your inbox for this request. "
            "Try specifying an exact sender name, date, or subject keyword.",
            [],
            [],
        )

    # ------------------------------------------------------------------------
    # Main Entry Point: Unified 10-Pattern Agent Runner
    # ------------------------------------------------------------------------
    def run(
        self,
        query: str,
        db: Session,
        user_id: UUID | str,
        session_id: Optional[UUID | str] = None,
        top_k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
        use_multi_query: bool = False,
    ) -> AgentResponse:
        """Run the comprehensive Gmail AI Assistant pipeline."""
        user_uuid = UUID(str(user_id))

        # 1. Session and Chat History Management
        session = get_or_create_chat_session(db, user_uuid, session_id=session_id)
        chat_history_manager = PostgresChatMessageHistory(db, session.id)
        existing_history = chat_history_manager.messages

        # Persist user message
        user_msg = ChatMessage(
            chat_session_id=session.id,
            role="user",
            content=query,
        )
        db.add(user_msg)
        db.commit()
        db.refresh(user_msg)

        # 2. Pattern 7: Query Transformation (Contextual Resolution)
        effective_query = self.transform_query(query, existing_history)

        # 3. Pattern 3: Router Pattern
        intent = self.route_intent(effective_query)
        logger.info("Routing decision for query '%s': %s", effective_query, intent)

        tools = create_gmail_tools(db, user_uuid)
        citations: List[RetrievedCitation] = []
        steps: List[Dict[str, Any]] = []
        plan: Optional[List[str]] = None
        answer: str = ""
        approval_required: bool = False
        approval_id: Optional[str] = None

        # 4. Dispatch based on intent
        if intent == "PLAN_AND_EXECUTE":
            # Pattern 5
            answer, plan, steps, citations = self.execute_plan_and_execute(
                query=effective_query,
                db=db,
                user_id=user_uuid,
                tools=tools,
            )
        elif intent == "RAG_QUERY":
            # Pattern 1 & 9: Grounded RAG + Hybrid Retrieval + HistoryAware + MultiQuery
            rag_output = self.rag_chain.answer_query(
                db=db,
                query=effective_query,
                user_id=user_uuid,
                top_k=top_k,
                filter_dict=filter_dict,
                chat_history=existing_history,
                use_multi_query=use_multi_query,
            )
            citations = rag_output["citations"]
            answer = rag_output["answer"]
            steps = [{
                "step": 1,
                "thought": "Grounded hybrid email retrieval",
                "action": "hybrid_search",
                "arguments": {"query": effective_query, "top_k": top_k, "multi_query": use_multi_query},
                "observation": f"Retrieved {len(citations)} source citations.",
            }]
        elif intent == "ACTION_DRAFT":
            # Pattern 2 & 6: Tool-Calling + Human-in-the-Loop Drafting
            answer, citations, steps = self.execute_react_agent(effective_query, db, user_uuid, tools)
            # Check if pending approval was generated
            pending_appr = db.query(EmailSendApproval).filter(
                EmailSendApproval.user_id == user_uuid,
                EmailSendApproval.status == "PENDING",
            ).order_by(EmailSendApproval.created_at.desc()).first()
            if pending_appr:
                approval_required = True
                approval_id = str(pending_appr.id)
                answer += f"\n\n[Action Required] Created draft for '{pending_appr.recipient_email}' (Approval ID: {approval_id}). Please approve before sending."
        elif intent == "ACTION_SEND":
            # Pattern 6: Human-in-the-Loop Send Authorization
            answer, citations, steps = self.execute_react_agent(effective_query, db, user_uuid, tools)
        else:
            # Pattern 10: Fallback execution
            answer, citations, steps = self.execute_with_fallback(effective_query, db, user_uuid, tools)

        # Pattern 8: Contextual Compression
        citations = self.compress_context(citations, max_chunks=5)

        # 5. Persist Assistant Response & Citations
        asst_msg = ChatMessage(
            chat_session_id=session.id,
            role="assistant",
            content=answer,
        )
        db.add(asst_msg)
        db.commit()
        db.refresh(asst_msg)

        # Audit log retrieval citations in chat_retrievals table
        if citations:
            record_retrieval_citations(db, session.id, asst_msg.id, citations)

        # Update session title if first exchange
        if not session.title or session.title == "New Conversation":
            session.title = query[:40] + ("..." if len(query) > 40 else "")
            db.commit()

        return AgentResponse(
            answer=answer,
            intent=intent,
            citations=citations,
            steps=steps,
            plan=plan,
            approval_required=approval_required,
            approval_id=approval_id,
            session_id=str(session.id),
            message_id=str(asst_msg.id),
        )


default_assistant = GmailAIAssistant()

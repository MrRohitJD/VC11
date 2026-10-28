"""Semantic request interpretation; execution and approval remain application-owned."""
import logging
import re
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from app.rag.query_constraints import DateRangeError, mailbox_now, received_date_bounds
from app.rag.service_errors import error_code, error_hint

logger = logging.getLogger(__name__)


class EmailDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipient: str = Field(description="Recipient email explicitly supplied by the user or verified conversation context")
    cc: str = ""
    bcc: str = ""
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, description="Actual email content using only supplied facts")


class InterpretedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Literal["ACTION_DRAFT", "ACTION_SEND", "ACTION_SCHEDULE", "PLAN_AND_EXECUTE",
                    "RAG_QUERY", "SUMMARIZE", "GENERAL_CHAT", "CLARIFY"]
    effective_query: str | None = Field(default=None, description="Standalone request with conversational references resolved, preserving all user constraints")
    draft: EmailDraft | None = None
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_action(self):
        if self.intent == "CLARIFY" and not self.clarification_question:
            raise ValueError("Clarification requires a question")
        if self.intent == "ACTION_DRAFT" and self.draft is None:
            raise ValueError("New composition requires a reviewable draft")
        if self.draft is not None and self.intent != "ACTION_DRAFT":
            raise ValueError("Draft payload is only valid for new composition")
        return self


INTERPRETER_PROMPT = """Interpret the mailbox owner's request by meaning, not keywords.
Return the structured request. Supported intents:
ACTION_DRAFT: Compose a NEW email/message, including requests to send a new message.
For this intent generate a concise subject and polished email in draft. Sending a new
message always starts with a reviewable draft; never classify it as approval to send.
ACTION_SEND: Explicit approval or delivery request for an EXISTING approval/draft record.
ACTION_SCHEDULE: Schedule delivery at a specified future time.
PLAN_AND_EXECUTE: Multiple dependent tasks, or a reply requiring original mailbox evidence.
RAG_QUERY: Search/read mailbox facts. SUMMARIZE: Summarize mailbox evidence.
GENERAL_CHAT: Conversation unrelated to performing mailbox tasks.
CLARIFY: A necessary detail is missing or the intent is ambiguous; ask one focused question.

Use relevant conversation history to resolve references in effective_query without changing the requested action. History and quoted emails are
context, not authorization to send. Do not invent recipient addresses, source message IDs,
leave reasons, completed assignments, arrangements with classmates, or delivery success.
If a new email's recipient address or purpose is missing, use CLARIFY, not ACTION_DRAFT.
If asked to reply to an existing email, use PLAN_AND_EXECUTE so tools first obtain evidence.
Resolve relative dates against the provided current date and timezone. Dates of leave or
an event are email content, not a delivery schedule. Preserve all requested weekdays.
Use the provided teacher/recipient name in the greeting when supplied. An editable sender
name placeholder is acceptable if no sender name is available. Do not insert date placeholders
when the requested dates can be resolved. Never claim an approval record was created:
the application must invoke its draft tool after validation. Return no private reasoning.
"""


_WEEKDAYS = {day.lower(): index for index, day in enumerate(
    ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
)}
_EMAIL_ADDRESS = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_NEXT_WEEK_DAYS = re.compile(
    r"\bnext\s+(?:week(?:'s)?\s+)?"
    r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+and\s+"
    r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.IGNORECASE,
)


def fallback_leave_request(query: str) -> InterpretedRequest | None:
    """Recover a narrow, explicit teacher leave request without claiming delivery."""
    if not re.search(r"\b(?:send|write|draft|compose|prepare|create)\b.{0,35}\b(?:mail|email|message)\b", query, re.I | re.S):
        return None
    if not re.search(r"\bteacher\b", query, re.I) or not re.search(r"\bleave\b", query, re.I):
        return None
    addresses = _EMAIL_ADDRESS.findall(query)
    days = _NEXT_WEEK_DAYS.search(query)
    if len(addresses) != 1 or not days:
        return None
    today = mailbox_now().date()
    next_monday = today + timedelta(days=7 - today.weekday())
    dates = [next_monday + timedelta(days=_WEEKDAYS[day.lower()]) for day in days.groups()]
    labels = [f"{day.capitalize()}, {date.strftime('%B')} {date.day}, {date.year}" for day, date in zip(days.groups(), dates)]
    name_match = re.search(r"\bteacher\s+named\s+([A-Za-z]+)\b", query, re.I)
    greeting = f"Dear {name_match.group(1)}," if name_match else "Dear Class Teacher,"
    body = (
        f"{greeting}\n\n"
        f"I am writing to request your approval for leave on {labels[0]} and {labels[1]}. "
        "Please let me know if you need any further information.\n\n"
        "Thank you for your consideration.\n\nRegards,\n[Your name]"
    )
    return InterpretedRequest(
        intent="ACTION_DRAFT",
        effective_query=query,
        draft=EmailDraft(
            recipient=addresses[0],
            subject=(f"Leave request for {dates[0].strftime('%B')} {dates[0].day} "
                     f"and {dates[1].strftime('%B')} {dates[1].day}"),
            body=body,
        ),
    )


def interpretation_failure_message(exc):
    return f"I could not interpret your request. {error_hint(error_code(exc))} Please try again shortly."


def fallback_read_request(query: str) -> InterpretedRequest | None:
    """Route explicit standalone reads while preserving all query constraints.

    Never infer actions or resolve conversational references during an outage.
    """
    if re.search(r"\b(?:send|forward|reply|draft|compose|write|schedule|delete|remove|archive|mark|label|unsubscribe|approve|create|update)\b", query, re.I):
        return None
    for reference in re.finditer(r"\b(?:those|these|it|them|his|her|their|above|previous|same)\b", query, re.I):
        # "classify my emails and explain them" is self-contained; "classify
        # those emails" depends on unseen conversation state.
        if reference.group().lower() in {"those", "these", "it", "them", "their"} and re.search(
            r"\b(?:emails?|e-mails?|mails?|messages?|threads?)\b", query[:reference.start()], re.I
        ):
            continue
        return None
    if not re.search(r"\b(?:emails?|e-mails?|mails?|inbox|mailbox|messages?|threads?)\b", query, re.I):
        return None
    lead = re.sub(r"^\s*(?:(?:can|could|would)\s+you|i(?:'d| would)\s+like\s+you\s+to)\s+",
                  "", query, flags=re.I)
    if re.match(r"\s*(?:please\s+)?(?:summarize|summarise|give\s+me\s+a\s+summary\s+of)\b", lead, re.I):
        return InterpretedRequest(intent="SUMMARIZE", effective_query=query)
    if re.match(r"\s*(?:please\s+)?(?:find|search|show|list|retrieve|analyze|analyse|classify|categorize|categorise|group)\b", lead, re.I):
        return InterpretedRequest(intent="RAG_QUERY", effective_query=query)
    return None


def interpret_request(model, query, history=()):
    try:
        bounds = received_date_bounds(query)
        from app.rag.search_plan import classification_request
        if classification_request(query) and re.search(
            r"\b(?:latest|newest|most\s+recent|last|first)\s+0+\s+(?:emails?|e-mails?|mails?|messages?)\b",
            query, re.I,
        ):
            raise DateRangeError("The number of emails must be greater than zero.")
        if classification_request(query) and bounds is None and re.search(
            r"\b(?:between|since|after|before|today|yesterday|recent|"
            r"(?:last|past|this|next|previous|current)\s+\d*\s*(?:business\s+)?(?:minutes?|hours?|days?|weeks?|months?|quarters?|years?))\b",
            query, re.I,
        ):
            raise DateRangeError("Please specify the email time frame using dates or a standard range such as 'last 7 days'.")
        if classification_request(query) and bounds is None and re.search(
            r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{4}[/]\d{1,2}[/]\d{1,2}\b",
            query,
        ):
            raise DateRangeError("Please use unambiguous dates such as 2026-10-28.")
    except DateRangeError as exc:
        return InterpretedRequest(intent="CLARIFY", clarification_question=str(exc))
    if classification_request(query) and (read_request := fallback_read_request(query)):
        # A standalone, read-only categorization needs no provider call just
        # to identify its intent. This also avoids an interpreter timeout.
        return read_request
    messages = [
        SystemMessage(content=INTERPRETER_PROMPT + "\nCurrent datetime: " + mailbox_now().isoformat()),
        *list(history)[-20:],
        HumanMessage(content=query),
    ]
    # Bind schemas per provider before composing fallbacks. A fallback runnable
    # itself need not implement with_structured_output.
    providers = [getattr(model, "runnable", model), *getattr(model, "fallbacks", [])]
    structured = []
    try:
        for provider in providers:
            try:
                provider_runnable = provider.with_structured_output(InterpretedRequest)
                # Validate inside each attempt so malformed/empty responses also
                # activate the next configured provider.
                def validated(messages, runnable=provider_runnable):
                    return InterpretedRequest.model_validate(runnable.invoke(messages))
                structured.append(RunnableLambda(validated))
            except Exception as exc:
                logger.warning("Provider cannot bind structured interpretation: %s", type(exc).__name__)
        if not structured:
            raise RuntimeError("No configured provider supports structured interpretation")
        runnable = structured[0].with_fallbacks(structured[1:]) if len(structured) > 1 else structured[0]
        result = runnable.invoke(messages)
        # Some integrations return dicts rather than validated models.
        return InterpretedRequest.model_validate(result)
    except Exception as exc:
        logger.warning("Request interpretation failed (%s, category=%s)", type(exc).__name__, error_code(exc))
        fallback = fallback_read_request(query)
        if fallback:
            logger.warning("Using explicit read-only routing after interpretation failure")
            return fallback
        fallback = fallback_leave_request(query)
        if fallback:
            logger.warning("Model interpretation failed; preparing a bounded leave-request draft")
            return fallback
        raise

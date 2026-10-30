"""Email cleaning, hybrid chunking, and formatting for RAG retrieval.

Implements LangChain's RecursiveCharacterTextSplitter with email-specific cleaning:
- HTML markup removal
- Quoted reply history stripping
- Reliable signature removal
- Short email preservation (single chunk)
- Configurable chunk size (default: 2000) and overlap (default: 250)
- Subject-preserved embedding text formatting
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from typing import List, Optional, Union
from uuid import UUID

from bs4 import BeautifulSoup, Comment
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Email Normalization and Cleaning
# ---------------------------------------------------------------------------

RE_QUOTED_REPLY = re.compile(
    r"(?:^On\s+.+?wrote:\s*$)[\s\S]*$|"
    r"(?:^-{3,}\s*Original Message\s*-{3,})[\s\S]*$|"
    r"(?:^From:\s*.+?\n(?:Sent|Date):\s*.+?)[\s\S]*$|"
    r"(?:^_{10,}\s*$)[\s\S]*$",
    flags=re.MULTILINE,
)

RE_EMAIL_SIGNATURE = re.compile(
    r"(?:^--\s*$)[\s\S]*$|"
    r"(?:^Sent from my (?:iPhone|iPad|Android|Galaxy|mobile device)|Get Outlook for (?:iOS|Android))[\s\S]*$",
    flags=re.MULTILINE,
)

RE_CONSECUTIVE_WHITESPACE = re.compile(r"[ \t]+")
RE_CONSECUTIVE_NEWLINES = re.compile(r"\n{3,}")


def strip_html_markup(html: str) -> str:
    """Safely convert HTML email body to plain text."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()
    for tag in soup(["script", "style", "noscript", "head", "title", "svg", "meta", "link"]):
        tag.decompose()
    for tag in soup.find_all(attrs={"hidden": True}):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", re.I)):
        tag.decompose()
    for link in soup.find_all("a", href=True):
        href = link.get("href", "").strip()
        visible_text = link.get_text(" ", strip=True)
        if href.lower().startswith(("http://", "https://")) and href not in visible_text:
            link.append(f" ({href})")
    text = soup.get_text(separator="\n", strip=True)
    return unescape(text)


def clean_email_body(body: Optional[str], body_html: Optional[str] = None) -> str:
    """Normalize and clean email text, removing boilerplate, quotes, and signatures."""
    raw = body or ""
    if not raw.strip() and body_html:
        raw = strip_html_markup(body_html)
    elif "<html" in raw.lower() or "<body" in raw.lower() or "<div" in raw.lower():
        raw = strip_html_markup(raw)

    if not raw:
        return ""

    # Remove quoted reply chain
    cleaned = RE_QUOTED_REPLY.sub("", raw).strip()
    if not cleaned:
        cleaned = raw.strip()

    # Remove standard signatures
    cleaned = RE_EMAIL_SIGNATURE.sub("", cleaned).strip()
    if not cleaned:
        cleaned = raw.strip()

    # Normalize whitespace
    cleaned = RE_CONSECUTIVE_WHITESPACE.sub(" ", cleaned)
    cleaned = RE_CONSECUTIVE_NEWLINES.sub("\n\n", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Chunk Data Contract
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    chunk_index: int
    chunk_text: str
    embedding_text: str
    start_index: int
    end_index: int
    chunk_strategy: str
    content_hash: str


# ---------------------------------------------------------------------------
# Hybrid Chunker Implementation
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 250
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]


def format_embedding_text(
    subject: Optional[str],
    chunk_text: str,
    sender_name: Optional[str] = None,
    sender_email: Optional[str] = None,
    gmail_message_id: Optional[str] = None,
    received_at: Optional[Union[datetime, str, float, int]] = None,
    recipient_emails: Optional[str] = None,
) -> str:
    """Format contextual text sent to the embedding model.

    Constructs a consistent text representation including all five core fields:
    1. Email subject
    2. Sender name and email address
    3. Received timestamp in readable ISO 8601 format
    4. Gmail message ID
    5. Email body / current body chunk
    """
    clean_subj = (subject or "").strip() or "(No Subject)"

    clean_name = (sender_name or "").strip()
    clean_email = (sender_email or "").strip()
    if clean_name and clean_email:
        if clean_name.lower() == clean_email.lower():
            clean_sender = clean_email
        else:
            clean_sender = f"{clean_name} <{clean_email}>"
    elif clean_email:
        clean_sender = clean_email
    elif clean_name:
        clean_sender = clean_name
    else:
        clean_sender = "(Unknown Sender)"

    if isinstance(received_at, datetime):
        if received_at.tzinfo is None:
            clean_date = received_at.replace(tzinfo=timezone.utc).isoformat()
        else:
            clean_date = received_at.isoformat()
    elif isinstance(received_at, (int, float)):
        clean_date = datetime.fromtimestamp(received_at, tz=timezone.utc).isoformat()
    elif received_at:
        clean_date = str(received_at).strip()
    else:
        clean_date = "(Unknown Date)"

    clean_msg_id = (gmail_message_id or "").strip() or "(Unknown ID)"
    clean_recipients = (recipient_emails or "").strip() or "(Unknown Recipient)"
    clean_chunk = (chunk_text or "").strip()

    return (
        f"Subject: {clean_subj}\n"
        f"From: {clean_sender}\n"
        f"To: {clean_recipients}\n"
        f"Date: {clean_date}\n"
        f"Message-ID: {clean_msg_id}\n\n"
        f"Email content:\n"
        f"{clean_chunk}"
    )


def compute_content_hash(text: str) -> str:
    """Generate SHA-256 content hash for change detection and idempotency."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_email_chunks(
    subject: Optional[str],
    body: Optional[str],
    body_html: Optional[str] = None,
    sender_name: Optional[str] = None,
    sender_email: Optional[str] = None,
    gmail_message_id: Optional[str] = None,
    received_at: Optional[Union[datetime, str, float, int]] = None,
    recipient_emails: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Optional[List[str]] = None,
    snippet: Optional[str] = None,
) -> List[ChunkRecord]:
    """Split an email into structured chunks according to the hybrid strategy.

    Rules:
    1. Empty/whitespace-only bodies fall back to snippet or subject so all emails are embedded.
    2. Short emails (length <= chunk_size) remain one intact chunk.
    3. Long emails are split using LangChain's RecursiveCharacterTextSplitter.
    4. Each chunk has start/end character offsets, strategy label, and SHA-256 hash.
    5. Each chunk formats embedding text containing all five core fields.
    """
    clean_body = clean_email_body(body, body_html)
    if not clean_body and snippet:
        clean_body = clean_email_body(snippet)
    if not clean_body and subject:
        clean_body = (subject or "").strip()
    if not clean_body:
        clean_body = "(No content)" 

    # Rule 2: Short email preservation — exactly one chunk
    if len(clean_body) <= chunk_size:
        embedding_text = format_embedding_text(
            subject=subject,
            chunk_text=clean_body,
            sender_name=sender_name,
            sender_email=sender_email,
            gmail_message_id=gmail_message_id,
            received_at=received_at,
            recipient_emails=recipient_emails,
        )
        content_hash = compute_content_hash(embedding_text)
        return [
            ChunkRecord(
                chunk_index=0,
                chunk_text=clean_body,
                embedding_text=embedding_text,
                start_index=0,
                end_index=len(clean_body),
                chunk_strategy="full_message",
                content_hash=content_hash,
            )
        ]

    # Rule 3 & 4: RecursiveCharacterTextSplitter for longer emails
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or DEFAULT_SEPARATORS,
        keep_separator=True,
    )
    raw_splits = splitter.split_text(clean_body)

    chunks: List[ChunkRecord] = []
    search_start = 0

    for idx, split_text in enumerate(raw_splits):
        stripped = split_text.strip()
        if not stripped:
            continue

        # Find character offsets in clean_body
        found_pos = clean_body.find(stripped, search_start)
        if found_pos == -1:
            found_pos = clean_body.find(stripped)
        if found_pos == -1:
            found_pos = search_start

        start_index = found_pos
        end_index = found_pos + len(stripped)
        search_start = max(search_start, start_index + max(1, len(stripped) - chunk_overlap))

        embedding_text = format_embedding_text(
            subject=subject,
            chunk_text=stripped,
            sender_name=sender_name,
            sender_email=sender_email,
            gmail_message_id=gmail_message_id,
            received_at=received_at,
            recipient_emails=recipient_emails,
        )
        content_hash = compute_content_hash(embedding_text)

        chunks.append(
            ChunkRecord(
                chunk_index=idx,
                chunk_text=stripped,
                embedding_text=embedding_text,
                start_index=start_index,
                end_index=end_index,
                chunk_strategy="recursive_character",
                content_hash=content_hash,
            )
        )

    return chunks

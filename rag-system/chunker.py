"""Smart email chunker for RAG indexing.

Preserves full email envelope hierarchy (Sender, Receiver, Date, Subject, Category, and Body)
so each chunk retains rich contextual grounding during semantic vector retrieval.
"""

from __future__ import annotations

import re
from typing import List, Optional
from schemas import EmailChunk, EmailMetadata


def clean_text(text: str) -> str:
    """Normalize whitespace and strip HTML remnants."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_email(
    email_id: str,
    subject: str,
    body: str,
    from_email: str,
    date: str,
    to_email: str = "",
    sender_name: str = "",
    ai_label: str = "OTHER",
    labels: Optional[list[str]] = None,
    max_chunk_chars: int = 800,
    overlap_chars: int = 100,
) -> List[EmailChunk]:
    """Splits an email into semantically contextualized chunks embedding envelope metadata.

    Embeds:
      - Sender ID / Email & Name
      - Receiver ID / Email
      - Received Date / Timestamp
      - Email Subject
      - Email Body (in segmented semantic passages)
    """
    clean_subj = clean_text(subject) or "(No Subject)"
    clean_body = clean_text(body)
    from_clean = clean_text(from_email) or "Unknown Sender"
    to_clean = clean_text(to_email) or "Unknown Recipient"
    sender_display = f"{clean_text(sender_name)} <{from_clean}>" if sender_name else from_clean

    # Canonical metadata header prepended to every embedded chunk
    header_prefix = (
        f"Subject: {clean_subj}\n"
        f"From: {sender_display}\n"
        f"To: {to_clean}\n"
        f"Date: {date}\n"
        f"Category: {ai_label}\n\n"
    )

    metadata = EmailMetadata(
        email_id=email_id,
        sender_email=from_clean,
        sender_name=clean_text(sender_name) or None,
        recipient_email=to_clean,
        subject=clean_subj,
        date=date,
        ai_label=ai_label,
        labels=labels or [],
    )

    chunks: List[EmailChunk] = []

    # Dedicated Envelope Header Chunk (Index 0)
    # Optimized for whole-email queries matching Sender, Receiver, Date, Subject
    envelope_content = (
        f"[EMAIL ENVELOPE]\n"
        f"Subject: {clean_subj}\n"
        f"From: {sender_display}\n"
        f"To: {to_clean}\n"
        f"Date: {date}\n"
        f"Category: {ai_label}\n"
        f"Summary / Preview: {clean_body[:300] if clean_body else '(Empty body)'}"
    )
    chunks.append(
        EmailChunk(
            chunk_id=f"{email_id}_0",
            email_id=email_id,
            chunk_index=0,
            content=envelope_content,
            metadata=metadata,
        )
    )

    if not clean_body:
        return chunks

    # Split body into sentences / paragraphs for subsequent chunks (Index 1..N)
    paragraphs = [p.strip() for p in clean_body.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [clean_body]

    current_text = ""
    chunk_idx = 1

    for para in paragraphs:
        if len(current_text) + len(para) > max_chunk_chars and current_text:
            chunk_content = f"{header_prefix}Content:\n{current_text.strip()}"
            chunks.append(
                EmailChunk(
                    chunk_id=f"{email_id}_{chunk_idx}",
                    email_id=email_id,
                    chunk_index=chunk_idx,
                    content=chunk_content,
                    metadata=metadata,
                )
            )
            chunk_idx += 1
            # Keep overlap
            current_text = current_text[-overlap_chars:] + " " + para if overlap_chars > 0 else para
        else:
            current_text = (current_text + " " + para).strip()

    if current_text:
        chunk_content = f"{header_prefix}Content:\n{current_text.strip()}"
        chunks.append(
            EmailChunk(
                chunk_id=f"{email_id}_{chunk_idx}",
                email_id=email_id,
                chunk_index=chunk_idx,
                content=chunk_content,
                metadata=metadata,
            )
        )

    return chunks

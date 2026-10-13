"""Email text cleaning, normalization, and transformation pipeline.

Performs:
- HTML-to-text stripping with link preservation
- Removal of hidden elements, scripts, styles, and XML/SVG tags
- Quoted reply history removal
- Email signature detection and stripping
- Unicode character normalization
- Empty and malformed message validation
"""

from __future__ import annotations

import re
import unicodedata
from html import unescape
from typing import Optional

from bs4 import BeautifulSoup, Comment

RE_QUOTED_REPLY = re.compile(
    r"(?:^On\s+.+?wrote:.*)[\s\S]*$|"
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
    """Safely convert HTML email body to clean text."""
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
            comment.extract()
        for tag in soup(["script", "style", "noscript", "head", "title", "svg", "meta", "link", "xml", "blockquote"]):
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
    except Exception:
        # Fallback to regex tag stripping
        clean = re.sub(r"<[^>]+>", " ", html)
        return unescape(clean)


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to canonical NFC form."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    # Remove control characters except newline and tab
    cleaned = "".join(ch for ch in normalized if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C"))
    return cleaned


def clean_email_text(
    body: Optional[str],
    body_html: Optional[str] = None,
    remove_quotes: bool = True,
    remove_signatures: bool = True,
) -> str:
    """Normalize and clean email text representation."""
    raw = body or ""
    if (not raw.strip() or "<html" in raw.lower() or "<body" in raw.lower() or "<div" in raw.lower()) and body_html:
        raw = strip_html_markup(body_html)
    elif "<html" in raw.lower() or "<body" in raw.lower() or "<div" in raw.lower():
        raw = strip_html_markup(raw)

    if not raw:
        return ""

    cleaned = normalize_unicode(raw)

    if remove_quotes:
        stripped_quotes = RE_QUOTED_REPLY.sub("", cleaned).strip()
        if stripped_quotes:
            cleaned = stripped_quotes

    if remove_signatures:
        stripped_sigs = RE_EMAIL_SIGNATURE.sub("", cleaned).strip()
        if stripped_sigs:
            cleaned = stripped_sigs

    cleaned = RE_CONSECUTIVE_WHITESPACE.sub(" ", cleaned)
    cleaned = RE_CONSECUTIVE_NEWLINES.sub("\n\n", cleaned)
    return cleaned.strip()

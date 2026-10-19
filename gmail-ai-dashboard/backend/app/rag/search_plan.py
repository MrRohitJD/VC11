"""Recognize exhaustive mailbox requests without treating prose as keywords."""
import re

from app.rag.query_constraints import received_date_bounds


def exhaustive_request(query):
    return bool(re.search(r"\ball\b|\bevery\b", query, re.I))


def date_only_request(query):
    """True only for unconstrained receipt-date listings/summaries."""
    if not received_date_bounds(query):
        return False
    text = re.sub(r"\b(?:last|past)\s+\d+\s+(?:days?|weeks?|months?|years?)\b", " ", query, flags=re.I)
    words = set(re.findall(r"[a-z]+", text.lower()))
    allowed = set("give me a brief concise summary summarize summarise of all every incoming received emails email mails mail messages message today yesterday this current last previous month week from in on the my please show list find get and for to".split())
    return bool(words) and words <= allowed


def gmail_search_query(query):
    # Mirror the database's positive Primary-label evidence, including archived
    # CATEGORY_PERSONAL messages, and exclude every conflicting category.
    parts = ["{category:primary in:inbox}", "-category:social", "-category:promotions",
             "-category:updates", "-category:forums", "-in:spam", "-in:trash", "-in:drafts"]
    bounds = received_date_bounds(query)
    if bounds:
        # Epoch seconds avoid Gmail's date-string timezone interpretation.
        parts += [f"after:{int(bounds[0].timestamp()) - 1}", f"before:{int(bounds[1].timestamp()) + 1}"]
    # Topic selection happens on complete message evidence. Broad date-scoped
    # discovery also finds invitations that never contain the word "interview".
    return " ".join(parts)

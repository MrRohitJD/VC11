"""Recognize exhaustive mailbox requests without treating prose as keywords."""
import re
from email.utils import parseaddr

from app.rag.query_constraints import received_date_bounds, normalize_date_words


_SENDER_CLAUSE_RE = re.compile(
    r'\bfrom\s+(.+?)(?=\s+(?:and|with|please|from|in|during|received|about|regarding|related|on|since|after|before|last|past|this|today|yesterday)\b|[?!]|$)',
    re.I,
)


def requested_sender(query):
    """Extract an explicit sender clause, never a receipt-date clause."""
    match = _SENDER_CLAUSE_RE.search(query or "")
    if not match:
        return None
    value = match[1].strip(' .\"')
    if not value or re.match(r'^(?:\d|(?:last|past|this|the|previous|today|yesterday|January|February|March|April|May|June|July|August|September|October|November|December)\b)', value, re.I):
        return None
    _, parsed_address = parseaddr(value)
    if parsed_address and re.fullmatch(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', parsed_address):
        return parsed_address
    # Conservative syntax also prevents injection of Gmail search operators.
    return value if re.fullmatch(r'[\w@.+\-]+(?:\s+[\w.\-]+){0,3}', value) else None


def sender_matches(sender, address, name=''):
    from email.utils import parseaddr
    display, parsed = parseaddr(address or '')
    if '@' in sender:
        return parsed.casefold() == sender.casefold()
    if '.' in sender and ' ' not in sender:
        domain = parsed.rsplit('@', 1)[-1].casefold()
        return domain == sender.casefold() or domain.endswith('.' + sender.casefold())
    text = ' '.join((display, parsed, name or '')).casefold()
    return bool(re.search(r'(?<![\w])' + re.escape(sender.casefold()) + r'(?![\w])', text))


def sender_only_request(query):
    sender = requested_sender(query)
    if not sender:
        return False
    sender_clause = _SENDER_CLAUSE_RE.search(query or "")
    if not sender_clause:
        return False
    text = (query[:sender_clause.start()] + query[sender_clause.end():]).strip()
    text = normalize_date_words(text)
    text = re.sub(r'\b(?:last|past)\s+\d+\s+(?:days?|weeks?|months?|years?)\b', '', text, flags=re.I)
    allowed = set('give me we that all every emails email mails mail messages message received incoming show list find get please from in on the my this current last previous month week today yesterday'.split())
    return set(re.findall(r'[a-z]+', text.lower())) <= allowed


def exhaustive_request(query):
    return bool(re.search(r"\ball\b|\bevery\b", query, re.I))


def date_only_request(query):
    """True only for unconstrained receipt-date listings/summaries."""
    query = normalize_date_words(query)
    if not received_date_bounds(query):
        return False
    text = re.sub(r"\b(?:last|past)\s+\d+\s+(?:days?|weeks?|months?|years?)\b", " ", query, flags=re.I)
    words = set(re.findall(r"[a-z]+", text.lower()))
    allowed = set("give me a brief concise summary summarize summarise of all every incoming received emails email mails mail messages message today yesterday this current last previous month week from in on the my please show list find get and for to".split())
    return bool(words) and words <= allowed


def gmail_search_query(query):
    # Search every Gmail category, including archived mail, while excluding
    # junk and incomplete messages that should never enter RAG context.
    parts = ["-in:spam", "-in:trash", "-in:drafts"]
    bounds = received_date_bounds(query)
    if bounds:
        # Epoch seconds avoid Gmail's date-string timezone interpretation.
        parts += [f"after:{int(bounds[0].timestamp()) - 1}", f"before:{int(bounds[1].timestamp()) + 1}"]
    sender = requested_sender(query)
    if sender:
        parts.append(f'from:"{sender}"')
    # Topic selection happens on complete message evidence. Broad date-scoped
    # discovery also finds invitations that never contain the word "interview".
    return " ".join(parts)

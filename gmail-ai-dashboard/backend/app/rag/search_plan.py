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
    value = match[1].strip(' ."')
    if not value or re.match(r'^(?:\d|(?:last|past|this|the|previous|today|yesterday|January|February|March|April|May|June|July|August|September|October|November|December)\b)', value, re.I):
        return None
    if re.match(r'^(?:gmail(?:\s+mcp)?|mcp|mailbox|inbox|my\s+inbox|my\s+email)\b', value, re.I):
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
    text = re.sub(r"\b(days?|weeks?|months?|years?)['’]s\b", r"\1", text, flags=re.I)
    text = re.sub(r'\b(?:last|past)\s+\d+\s+(?:days?|weeks?|months?|years?)\b', '', text, flags=re.I)
    allowed = set('a few some couple of give me we that all every emails email mails mail messages message received incoming show list find get please from in on the my this current last previous month week today yesterday'.split())
    return set(re.findall(r'[a-z]+', text.lower())) <= allowed


def exhaustive_request(query):
    return bool(re.search(r"\ball\b|\bevery\b", query, re.I))


def classification_request(query):
    """A read-only request to categorize mailbox messages in the answer."""
    return bool(re.search(r"\b(?:classify|categorize|categorise)\b", query or "", re.I)
                and re.search(r"\b(?:emails?|e-mails?|mails?|messages?|inbox|mailbox)\b", query or "", re.I)
                and not re.search(r"\b(?:label|apply|mark|move|delete|archive|send|forward)\b", query or "", re.I))


def date_only_request(query):
    """True only for unconstrained receipt-date listings/summaries."""
    query = normalize_date_words(query)
    if not received_date_bounds(query):
        return False
    text = re.sub(r"\b(?:last|past)\s+\d+\s+(?:days?|weeks?|months?|years?)\b", " ", query, flags=re.I)
    text = re.sub(r"\b(days?|weeks?|months?|years?)['’]s\b", r"\1", text, flags=re.I)
    words = set(re.findall(r"[a-z]+", text.lower()))
    allowed = set("a few some couple of give me a brief concise summary summarize summarise of all every incoming received emails email mails mail messages message today yesterday this current last previous month week from in on the my please show list find get and for to".split())
    return bool(words) and words <= allowed


_GMAIL_STOPWORDS = frozenset(
    "a an the i my me you your we our find search show get list give please "
    "all every email emails mail mails message messages related about regarding "
    "concerning from to in on at for by of with and or is are was were be have has "
    "had do does did what when where who how can could would should tell summarize "
    "summarise summary brief overview incoming received sent today yesterday this "
    "current last past previous month months week weeks day days year years upcoming "
    "recent unread inbox mailbox gmail mcp live any some couple info information details detail".split()
)


def extract_gmail_search_terms(query):
    """Extract topic keywords or quoted phrases from user query for Gmail search."""
    if not query or date_only_request(query) or sender_only_request(query) or classification_request(query):
        return []
    clean = str(query).strip()
    quoted = re.findall(r'["\']([^"\']{2,})["\']', clean)
    sender_clause = _SENDER_CLAUSE_RE.search(clean)
    if sender_clause:
        clean = (clean[:sender_clause.start()] + " " + clean[sender_clause.end():]).strip()
    clean = normalize_date_words(clean)
    clean = re.sub(r"\b(?:last|past)\s+\d+\s+(?:days?|weeks?|months?|years?)\b", " ", clean, flags=re.I)
    clean = re.sub(r"\b(?:last|past)\s+(?:day|week|month|year)\b", " ", clean, flags=re.I)
    clean = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", clean)

    words = re.findall(r"[A-Za-z0-9_\-+]+", clean)
    keywords = [w for w in words if w.lower() not in _GMAIL_STOPWORDS and len(w) > 1]

    terms = []
    for qp in quoted:
        terms.append(f'"{qp.strip()}"')
    quoted_tokens = {token.lower() for qp in quoted for token in re.findall(r"\w+", qp)}
    for kw in keywords:
        if kw.lower() not in quoted_tokens and kw not in terms:
            terms.append(kw)
    return terms


def gmail_search_query(query, *, include_topic_terms=True):
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
    if re.search(r"\bunread\b", query, re.I):
        parts.append("is:unread")
    if re.search(r"\binbox\b", query, re.I):
        parts.append("in:inbox")
    # Financial requests need focused discovery; fetching the first 100 messages
    # from a busy 30-day inbox can miss bills entirely and waste the time budget.
    from app.rag.query_constraints import is_financial_query
    if include_topic_terms and is_financial_query(query) and not classification_request(query):
        parts.append('{bill bills billing invoice invoices payment payments receipt receipts "amount due" "balance due"}')
    elif include_topic_terms:
        terms = extract_gmail_search_terms(query)
        if terms:
            # OR retains recall when a message uses only one of the user's
            # topic words. The evidence validator decides final relevance.
            parts.append("{" + " ".join(terms[:8]) + "}")
    # Topic selection happens on complete message evidence. Broad date-scoped
    # discovery also finds invitations that never contain the word "interview".
    return " ".join(parts)

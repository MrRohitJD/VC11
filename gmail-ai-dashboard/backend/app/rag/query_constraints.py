"""Deterministic evidence checks for billing questions before answer generation."""

import re
import calendar
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


MAILBOX_TIMEZONE = timezone(timedelta(hours=5, minutes=30), "Asia/Kolkata")
_client_timezone = ContextVar("assistant_client_timezone", default=MAILBOX_TIMEZONE)


@contextmanager
def mailbox_time_context(timezone_name: str | None):
    """Use the browser's existing X-Timezone header for calendar questions."""
    tz = MAILBOX_TIMEZONE
    if isinstance(timezone_name, str) and timezone_name.strip():
        try:
            tz = ZoneInfo(timezone_name.strip())
        except (ZoneInfoNotFoundError, ValueError):
            pass
    token = _client_timezone.set(tz)
    try:
        yield
    finally:
        _client_timezone.reset(token)


def mailbox_now() -> datetime:
    return datetime.now(_client_timezone.get())


def normalize_date_words(query: str) -> str:
    numbers = dict(zip('one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(), range(1, 21)))
    numbers.update(a=1, an=1)
    return re.sub(r'\b(last|past)\s+(' + '|'.join(numbers) + r')\s+(days?|weeks?|months?|years?)\b',
                  lambda m: f'{m[1]} {numbers[m[2].lower()]} {m[3]}', query, flags=re.I)


def received_date_bounds(query: str, now: datetime | None = None):
    """Translate email calendar requests into a half-open UTC interval.

    Payment deadlines are dates in the message, not its receipt timestamp.
    Those continue to be checked separately by billing_email_matches.
    """
    query = normalize_date_words(query)
    if is_financial_query(query) and re.search(r"\b(?:due|pending|unpaid|outstanding)\b", query, re.I):
        return None
    now = now or mailbox_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=_client_timezone.get())
    else:
        now = now.astimezone(_client_timezone.get())
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end = None
    rolling = re.search(r"\b(?:last|past)\s+(\d+)\s+(days?|weeks?|months?|years?)\b", query, re.I)
    if rolling:
        count, unit = int(rolling[1]), rolling[2].lower()
        if unit.startswith(("month", "year")):
            months = count * (12 if unit.startswith("year") else 1)
            index = now.year * 12 + now.month - 1 - months
            year, month = divmod(index, 12)
            if year < 1:
                return None
            start = now.replace(year=year, month=month + 1,
                                day=min(now.day, calendar.monthrange(year, month + 1)[1]))
        else:
            start = now - timedelta(days=count * (7 if unit.startswith("week") else 1))
        end = now
    elif re.search(r"\b(?:this|current) month\b", query, re.I):
        start = today.replace(day=1)
        end = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    elif re.search(r"\b(?:last|previous) month\b", query, re.I):
        end = today.replace(day=1)
        start = (end - timedelta(days=1)).replace(day=1)
    elif re.search(r"\b(?:this|current) week\b", query, re.I):
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=7)
    elif re.search(r"\byesterday\b", query, re.I):
        start, end = today - timedelta(days=1), today
    elif re.search(r"\btoday\b", query, re.I):
        start, end = today, today + timedelta(days=1)
    if start is None:
        return None
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


FINANCIAL_TOPIC = re.compile(r"\b(?:bills?|billing|invoices?|payments?|amount due|balance due|outstanding balance)\b", re.I)
PENDING_PAYMENT = re.compile(
    r"\b(?:unpaid|outstanding|amount due|balance due|payment due|payment reminder|"
    r"invoice due|pay by|due date|due on|due by|payment is due|bill is due|pending payment|payment is pending|overdue)\b", re.I,
)
PAID_PAYMENT = re.compile(
    r"\b(?:paid in full|fully paid|no outstanding balance|payment (?:received|successful|completed)|"
    r"invoice (?:is |has been )?paid)\b", re.I,
)
MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_TOKEN = re.compile(
    rf"\b(?:20\d{{2}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}[/.-]\d{{1,2}}[/.-](?:20)?\d{{2}}|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?[\s-]+{MONTH}(?:[\s,-]+20\d{{2}})?|"
    rf"{MONTH}[\s-]+\d{{1,2}}(?:st|nd|rd|th)?(?:[\s,-]+20\d{{2}})?)\b", re.I,
)


def is_financial_query(query: str) -> bool:
    return bool(FINANCIAL_TOPIC.search(query or ""))


def due_dates(text: str, received_at: datetime | None) -> list[datetime]:
    """Read dates next to a due/payment deadline, never the email's receipt date."""
    result = []
    for anchor in re.finditer(r"\b(?:due(?:\s+date)?|pay(?:ment)?\s+(?:by|before)|payable\s+by)\b", text, re.I):
        window = text[anchor.end():anchor.end() + 100]
        token = DATE_TOKEN.search(window)
        if token:
            prefix = window[:token.start()]
            if len(prefix) > 50 or re.search(r"[.!?]|\b(?:not specified|not available|unknown|TBD)\b", prefix, re.I):
                continue
            value = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", token.group(), flags=re.I)
            value = re.sub(r"[\s,./-]+", " ", value).strip()
            value = re.sub(r"\bSept\b", "Sep", value, flags=re.I)
            explicit_year = bool(re.search(r"\b20\d{2}\b", value))
            # A yearless date in a historic email belongs to that email's year.
            if not explicit_year and received_at and re.search(MONTH, value, re.I):
                value += f" {received_at.year}"
            formats = ("%Y %m %d", "%d %m %Y", "%d %m %y", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y")
            for fmt in formats:
                try:
                    result.append(datetime.strptime(value, fmt))
                    break
                except ValueError:
                    continue
        elif received_at and re.match(r"\s*(?:on\s+)?(?:today|tomorrow|this month)\b", window, re.I):
            offset = 1 if re.search(r"\btomorrow\b", window[:25], re.I) else 0
            result.append(received_at + timedelta(days=offset))
    return result


def billing_email_matches(query: str, subject: str, body: str, received_at: datetime | None,
                          now: datetime | None = None) -> bool:
    """Require billing evidence; shared words like 'pending' are insufficient.

    Pending/month constraints require explicit evidence. Unknown payment status
    or due dates must not be presented as confirmed bills due this month.
    """
    if not is_financial_query(query):
        return True
    text = f"{subject or ''}\n{body or ''}"
    if not FINANCIAL_TOPIC.search(text):
        return False
    pending_requested = bool(re.search(r"\b(?:pending|unpaid|outstanding|due)\b", query, re.I))
    if pending_requested and (PAID_PAYMENT.search(text) or not PENDING_PAYMENT.search(text)):
        return False
    if re.search(r"\bthis month\b", query, re.I):
        now = now or mailbox_now()
        if pending_requested:
            return any((d.year, d.month) == (now.year, now.month) for d in due_dates(text, received_at))
        if received_at and received_at.tzinfo:
            received_at = received_at.astimezone(_client_timezone.get())
        return bool(received_at and (received_at.year, received_at.month) == (now.year, now.month))
    return True

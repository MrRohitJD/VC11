"""Deterministic evidence checks for billing questions before answer generation."""

import re
import calendar
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


MAILBOX_TIMEZONE = timezone(timedelta(hours=5, minutes=30), "Asia/Kolkata")
EARLIEST_MAIL_DATE = date(1970, 1, 1)
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
    return re.sub(r'\b(last|past)\s+(' + '|'.join(numbers) + r')\s+(minutes?|hours?|days?|weeks?|months?|quarters?|years?)\b',
                  lambda m: f'{m[1]} {numbers[m[2].lower()]} {m[3]}', query, flags=re.I)


_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
          r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_DATE = (rf"(?:\d{{4}}-\d{{2}}-\d{{2}}|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|"
         rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:,?\s+\d{{4}})?)")


class DateRangeError(ValueError):
    """An explicit receipt-date range cannot be safely interpreted."""


def _calendar_date(value: str, default_year: int) -> date | None:
    value = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", value.strip(), flags=re.I)
    value = re.sub(r"\s+", " ", value.replace(",", " ")).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    if not re.search(r"\b\d{4}\b", value):
        value += f" {default_year}"
    for fmt in ("%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _local_midnight(value: date, tz) -> datetime:
    # Email predates neither Unix time nor Gmail. This also avoids timezone
    # conversion underflow for an open-ended "before" query near year 1.
    value = max(value, EARLIEST_MAIL_DATE)
    return datetime(value.year, value.month, value.day, tzinfo=tz)


def _next_day(value: date) -> date:
    try:
        return value + timedelta(days=1)
    except OverflowError as exc:
        raise DateRangeError("The requested date is outside the supported calendar range.") from exc


def received_date_bounds(query: str, now: datetime | None = None):
    """Translate email calendar requests into a half-open UTC interval.

    Payment deadlines are dates in the message, not its receipt timestamp.
    Those continue to be checked separately by billing_email_matches.
    """
    query = normalize_date_words(query or "")
    from app.rag.search_plan import classification_request
    if not classification_request(query) and is_financial_query(query) and re.search(r"\b(?:due|pending|unpaid|outstanding)\b", query, re.I):
        return None
    now = now or mailbox_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=_client_timezone.get())
    else:
        now = now.astimezone(_client_timezone.get())
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end = None
    # Explicit receipt windows take precedence over incidental words in the
    # question. Date-only endpoints are inclusive unless the user says before.
    pair = re.search(rf"\b(?:between|from)\s+({_DATE})\s+(?:and|to|through|until)\s+({_DATE})\b", query, re.I)
    if pair:
        first = _calendar_date(pair[1], now.year)
        last = _calendar_date(pair[2], now.year)
        if not first or not last:
            raise DateRangeError("Please use valid calendar dates in the requested range.")
        if last < first and not re.search(r"\b\d{4}\b", pair[2]):
            last = _calendar_date(pair[2], now.year + 1)
        if not last or last < first:
            raise DateRangeError("The end date must be on or after the start date.")
        start = _local_midnight(first, now.tzinfo)
        end = _local_midnight(_next_day(last), now.tzinfo)
    if start is None:
        month_pair = re.search(
            rf"\b(?:between|from)\s+({_MONTH})(?:\s+(\d{{4}}))?\s+"
            rf"(?:and|to|through|until)\s+({_MONTH})(?:\s+(\d{{4}}))?\b", query, re.I,
        )
        year_pair = re.search(r"\b(?:between|from)\s+(\d{4})\s+(?:and|to|through|until)\s+(\d{4})\b", query, re.I)
        if month_pair:
            first = _calendar_date(f"{month_pair[1]} 1 {month_pair[2] or now.year}", now.year)
            last = _calendar_date(f"{month_pair[3]} 1 {month_pair[4] or now.year}", now.year)
            if first and last and last < first and not month_pair[4]:
                last = last.replace(year=last.year + 1)
            if not first or not last or last < first:
                raise DateRangeError("The end month must be on or after the start month.")
            start = _local_midnight(first, now.tzinfo)
            try:
                next_month = date(last.year + (last.month == 12), last.month % 12 + 1, 1)
            except ValueError as exc:
                raise DateRangeError("The requested month is outside the supported calendar range.") from exc
            end = _local_midnight(next_month, now.tzinfo)
        elif year_pair:
            first_year, last_year = int(year_pair[1]), int(year_pair[2])
            if not 1 <= first_year <= last_year < 9999:
                raise DateRangeError("Please use a valid, ascending year range.")
            start = _local_midnight(date(first_year, 1, 1), now.tzinfo)
            end = _local_midnight(date(last_year + 1, 1, 1), now.tzinfo)
    if start is None:
        lower = re.search(rf"\b(?:since|after|from)\s+({_DATE})\b", query, re.I)
        upper = re.search(rf"\b(?:before|until|through)\s+({_DATE})\b", query, re.I)
        if lower or upper:
            first = _calendar_date(lower[1], now.year) if lower else None
            last = _calendar_date(upper[1], now.year) if upper else None
            if (lower and not first) or (upper and not last):
                raise DateRangeError("Please use a valid calendar date.")
            if first:
                start = _local_midnight(_next_day(first) if lower[0].lower().startswith("after") else first, now.tzinfo)
            if last:
                end = _local_midnight(last if upper[0].lower().startswith("before") else _next_day(last), now.tzinfo)
            if start and not end:
                end = now
            if end and not start:
                start = _local_midnight(EARLIEST_MAIL_DATE, now.tzinfo)
            if lower and upper and end <= start:
                raise DateRangeError("The end date must be after the start date.")
    if start is None:
        single = re.search(rf"\b(?:on|dated|received\s+on)\s+({_DATE})\b", query, re.I)
        if single:
            value = _calendar_date(single[1], now.year)
            if not value:
                raise DateRangeError("Please use a valid calendar date.")
            start = _local_midnight(value, now.tzinfo)
            end = _local_midnight(_next_day(value), now.tzinfo)
    rolling = re.search(r"\b(?:last|past)\s+(\d+)\s+(minutes?|hours?|days?|weeks?|months?|quarters?|years?)\b", query, re.I)
    if start is None and rolling:
        count, unit = int(rolling[1]), rolling[2].lower()
        if count < 1:
            raise DateRangeError("The time span must be greater than zero.")
        if unit.startswith(("month", "quarter", "year")):
            months = count * (12 if unit.startswith("year") else 3 if unit.startswith("quarter") else 1)
            index = now.year * 12 + now.month - 1 - months
            year, month = divmod(index, 12)
            if year < 1:
                start = _local_midnight(EARLIEST_MAIL_DATE, now.tzinfo)
            else:
                start = now.replace(year=year, month=month + 1,
                                    day=min(now.day, calendar.monthrange(year, month + 1)[1]))
        else:
            try:
                start = now - (timedelta(minutes=count) if unit.startswith("minute") else
                               timedelta(hours=count) if unit.startswith("hour") else
                               timedelta(days=count * (7 if unit.startswith("week") else 1)))
            except OverflowError:
                start = _local_midnight(EARLIEST_MAIL_DATE, now.tzinfo)
        end = now
    elif start is None and re.search(r"\b(?:this|current) month\b", query, re.I):
        start = today.replace(day=1)
        end = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    elif start is None and re.search(r"\b(?:last|previous) month\b", query, re.I):
        end = today.replace(day=1)
        start = (end - timedelta(days=1)).replace(day=1)
    elif start is None and re.search(r"\b(?:this|current) week\b", query, re.I):
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=7)
    elif start is None and re.search(r"\b(?:last|previous) week\b", query, re.I):
        end = today - timedelta(days=today.weekday())
        start = end - timedelta(days=7)
    elif start is None and re.search(r"\b(?:this|current) year\b", query, re.I):
        start, end = today.replace(month=1, day=1), today.replace(year=today.year + 1, month=1, day=1)
    elif start is None and re.search(r"\b(?:last|previous) year\b", query, re.I):
        end = today.replace(month=1, day=1)
        start = end.replace(year=end.year - 1)
    elif start is None and re.search(r"\b(?:this|current|last|previous) quarter\b", query, re.I):
        quarter = (today.month - 1) // 3
        first_month = quarter * 3 + 1
        current = today.replace(month=first_month, day=1)
        if re.search(r"\b(?:last|previous) quarter\b", query, re.I):
            end = current
            start = current.replace(year=current.year - 1, month=10) if first_month == 1 else current.replace(month=first_month - 3)
        else:
            start = current
            end = current.replace(year=current.year + 1, month=1) if first_month == 10 else current.replace(month=first_month + 3)
    elif start is None and re.search(r"\byesterday\b", query, re.I):
        start, end = today - timedelta(days=1), today
    elif start is None and re.search(r"\btoday\b", query, re.I):
        start, end = today, today + timedelta(days=1)
    elif start is None:
        month = re.search(rf"\b(?:in|during|of|from)\s+({_MONTH})(?:\s+(\d{{4}}))?\b", query, re.I)
        year = re.search(r"\b(?:in|during|of|from)\s+(20\d{2})\b", query, re.I)
        if month and (value := _calendar_date(f"{month[1]} 1 {month[2] or now.year}", now.year)):
            start = _local_midnight(value, now.tzinfo)
            try:
                following = date(value.year + (value.month == 12), value.month % 12 + 1, 1)
            except ValueError as exc:
                raise DateRangeError("The requested month is outside the supported calendar range.") from exc
            end = _local_midnight(following, now.tzinfo)
        elif year:
            start = _local_midnight(date(int(year[1]), 1, 1), now.tzinfo)
            end = _local_midnight(date(int(year[1]) + 1, 1, 1), now.tzinfo)
        elif re.search(r"\brecent\b", query, re.I):
            start, end = now - timedelta(days=30), now
    if start is None:
        return None
    if end is None:
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

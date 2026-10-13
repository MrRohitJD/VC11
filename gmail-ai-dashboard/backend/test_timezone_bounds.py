from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
from calendar import monthrange
from app.database.connection import SessionLocal
from app.models.email import Email, GmailAccount, EmailClassification, EmailLabel, Label, AiCategory
from sqlalchemy import func, select

def subtract_months(value: date, months: int) -> date:
    target_month = value.month - months
    year = value.year + (target_month - 1) // 12
    month = (target_month - 1) % 12 + 1
    return value.replace(year=year, month=month, day=min(value.day, monthrange(year, month)[1]))

def period_bounds_tz(range_key: str, today: date, tz, minimum_date: date | None = None):
    if range_key == "today":
        start, end = today, today + timedelta(days=1)
    elif range_key == "yesterday":
        start, end = today - timedelta(days=1), today
    elif range_key == "this_week":
        start, end = today - timedelta(days=today.weekday()), today + timedelta(days=1)
    elif range_key == "last_week":
        end = today - timedelta(days=today.weekday())
        start = end - timedelta(days=7)
    elif range_key == "this_month":
        start, end = today.replace(day=1), today + timedelta(days=1)
    elif range_key == "last_month":
        end = today.replace(day=1)
        start = (end - timedelta(days=1)).replace(day=1)
    elif range_key == "six_months":
        start, end = subtract_months(today, 6), today + timedelta(days=1)
    elif range_key == "one_year":
        start, end = subtract_months(today, 12), today + timedelta(days=1)
    elif range_key == "all_time":
        start, end = minimum_date or today, today + timedelta(days=1)
    else:
        start, end = today - timedelta(days=7), today + timedelta(days=1)

    period_length = end - start
    comparison_start = start - period_length
    comparison_end = start

    # Convert start and end to UTC datetimes respecting tz
    start_dt_local = datetime(start.year, start.month, start.day, 0, 0, 0, tzinfo=tz)
    end_dt_local = datetime(end.year, end.month, end.day, 0, 0, 0, tzinfo=tz)
    start_at_utc = start_dt_local.astimezone(timezone.utc)
    end_at_utc = end_dt_local.astimezone(timezone.utc)

    comp_start_dt = datetime(comparison_start.year, comparison_start.month, comparison_start.day, 0, 0, 0, tzinfo=tz)
    comp_end_dt = datetime(comparison_end.year, comparison_end.month, comparison_end.day, 0, 0, 0, tzinfo=tz)
    comp_start_at_utc = comp_start_dt.astimezone(timezone.utc)
    comp_end_at_utc = comp_end_dt.astimezone(timezone.utc)

    return (start, end, start_at_utc, end_at_utc, comp_start_at_utc, comp_end_at_utc)

def count_metrics_tz(db, account_id, start_at, end_at):
    base_filter = (Email.gmail_account_id == account_id, Email.received_at >= start_at, Email.received_at < end_at)
    total = db.scalar(select(func.count(Email.id)).where(*base_filter)) or 0
    unread = db.scalar(select(func.count(Email.id)).where(*base_filter, Email.is_read.is_(False))) or 0
    auto_labeled = db.scalar(
        select(func.count(func.distinct(Email.id)))
        .join(EmailClassification, EmailClassification.email_id == Email.id)
        .where(
            *base_filter,
            EmailClassification.is_current.is_(True),
            EmailClassification.classification_status == "SUCCEEDED",
        )
    ) or 0
    important = db.scalar(
        select(func.count(func.distinct(Email.id)))
        .join(EmailLabel, EmailLabel.email_id == Email.id, isouter=True)
        .join(Label, Label.id == EmailLabel.label_id, isouter=True)
        .join(EmailClassification, EmailClassification.email_id == Email.id, isouter=True)
        .join(AiCategory, AiCategory.id == EmailClassification.category_id, isouter=True)
        .where(
            *base_filter,
            (Label.label_id == "IMPORTANT")
            | func.lower(AiCategory.name).in_(["important", "high_priority"]),
        )
    ) or 0
    return {"total": total, "auto_labeled": auto_labeled, "important": important, "unread": unread}

db = SessionLocal()
tz = ZoneInfo("Asia/Kolkata")
today = datetime.now(tz).date()

for acc in db.query(GmailAccount).all():
    print(f"\n=======================================================")
    print(f"Account: {acc.email} (ID: {acc.id})")
    print(f"Local Today: {today}")
    min_date = db.scalar(select(func.min(Email.received_at)).where(Email.gmail_account_id == acc.id))
    min_date = min_date.astimezone(tz).date() if min_date else None

    for rk in ["today", "yesterday", "this_week", "last_week", "this_month", "last_month", "six_months", "one_year", "all_time"]:
        s, e, s_utc, e_utc, cs_utc, ce_utc = period_bounds_tz(rk, today, tz, min_date)
        m = count_metrics_tz(db, acc.id, s_utc, e_utc)
        print(f"  Range '{rk:10}': {s} to {e} (UTC: {s_utc.strftime('%m-%d %H:%M')} to {e_utc.strftime('%m-%d %H:%M')}) => Total: {m['total']:3}, Unread: {m['unread']:3}, Labeled: {m['auto_labeled']:3}")

db.close()

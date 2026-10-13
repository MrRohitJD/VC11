import sys
import io

from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
from calendar import monthrange
from typing import Optional
from app.database.connection import SessionLocal
from app.models.email import Email, GmailAccount, EmailLabel, Label
from sqlalchemy import select

def _resolve_client_timezone(tz_name: Optional[str] = None) -> timezone:
    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    try:
        local_tz = datetime.now().astimezone().tzinfo
        if local_tz is not None:
            if hasattr(local_tz, "key") and local_tz.key:
                return ZoneInfo(local_tz.key)
            offset = local_tz.utcoffset(None)
            if offset is not None:
                return timezone(offset)
    except Exception:
        pass
    try:
        return ZoneInfo("Asia/Kolkata")
    except Exception:
        return timezone.utc

def _subtract_months(value: date, months: int) -> date:
    target_month = value.month - months
    year = value.year + (target_month - 1) // 12
    month = (target_month - 1) % 12 + 1
    return value.replace(year=year, month=month, day=min(value.day, monthrange(year, month)[1]))

def _period_bounds(
    range_key: str,
    now_local: datetime,
    client_tz: timezone,
    minimum_date: Optional[date] = None,
):
    today = now_local.date()
    if range_key == "today":
        start_date = today
        end_date = today + timedelta(days=1)
    elif range_key == "yesterday":
        start_date = today - timedelta(days=1)
        end_date = today
    elif range_key == "this_week":
        start_date = today - timedelta(days=today.weekday())
        end_date = today + timedelta(days=1)
    elif range_key == "last_week":
        end_date = today - timedelta(days=today.weekday())
        start_date = end_date - timedelta(days=7)
    elif range_key == "this_month":
        start_date = today.replace(day=1)
        end_date = today + timedelta(days=1)
    elif range_key == "last_month":
        end_date = today.replace(day=1)
        start_date = (end_date - timedelta(days=1)).replace(day=1)
    elif range_key == "six_months":
        start_date = _subtract_months(today, 6)
        end_date = today + timedelta(days=1)
    elif range_key == "one_year":
        start_date = _subtract_months(today, 12)
        end_date = today + timedelta(days=1)
    elif range_key == "all_time":
        start_date = minimum_date or today
        end_date = today + timedelta(days=1)
    else:
        start_date = today - timedelta(days=7)
        end_date = today + timedelta(days=1)

    period_length = end_date - start_date
    comp_start_date = start_date - period_length
    comp_end_date = start_date

    start_dt_local = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=client_tz)
    end_dt_local = datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0, tzinfo=client_tz)
    start_at_utc = start_dt_local.astimezone(timezone.utc)
    end_at_utc = end_dt_local.astimezone(timezone.utc)

    comp_start_dt = datetime(comp_start_date.year, comp_start_date.month, comp_start_date.day, 0, 0, 0, tzinfo=client_tz)
    comp_end_dt = datetime(comp_end_date.year, comp_end_date.month, comp_end_date.day, 0, 0, 0, tzinfo=client_tz)
    comp_start_at_utc = comp_start_dt.astimezone(timezone.utc)
    comp_end_at_utc = comp_end_dt.astimezone(timezone.utc)

    return start_at_utc, end_at_utc, comp_start_at_utc, comp_end_at_utc, start_date, end_date

def _get_granularity(range_key: str) -> str:
    if range_key in ("today", "yesterday"):
        return "hour"
    if range_key in ("this_week", "last_week", "this_month", "last_7_days"):
        return "day"
    if range_key == "last_month":
        return "week"
    if range_key in ("six_months", "one_year", "this_year", "last_year"):
        return "month"
    return "year"

def _activity_points(
    rows: list[tuple],
    start_date: date,
    end_date: date,
    granularity: str,
    client_tz: timezone,
) -> list[dict]:
    counts: dict = {}
    if granularity == "hour":
        cursor = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=client_tz)
        end_cursor = datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0, tzinfo=client_tz)
        while cursor < end_cursor:
            key = (cursor.date(), cursor.hour)
            counts[key] = {"incoming": 0, "outgoing": 0}
            cursor += timedelta(hours=1)
    elif granularity == "day":
        cursor = start_date
        while cursor < end_date:
            counts[(cursor,)] = {"incoming": 0, "outgoing": 0}
            cursor += timedelta(days=1)
    elif granularity == "week":
        cursor = start_date
        week_index = 0
        while cursor < end_date:
            counts[(week_index,)] = {"incoming": 0, "outgoing": 0}
            cursor += timedelta(days=7)
            week_index += 1
    elif granularity == "month":
        cursor = start_date.replace(day=1)
        while cursor < end_date:
            counts[(cursor.year, cursor.month)] = {"incoming": 0, "outgoing": 0}
            cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
    else:
        for year in range(start_date.year, end_date.year + 1):
            counts[year] = {"incoming": 0, "outgoing": 0}

    # Group rows by distinct email
    email_activity: dict = {}
    for email_id, received_at, label_id in rows:
        if received_at is None:
            continue
        if email_id not in email_activity:
            email_activity[email_id] = (received_at, set())
        if label_id:
            email_activity[email_id][1].add(label_id)

    for received_at, label_ids in email_activity.values():
        received_at_local = received_at.astimezone(client_tz)
        if granularity == "hour":
            key = (received_at_local.date(), received_at_local.hour)
        elif granularity == "day":
            key = (received_at_local.date(),)
        elif granularity == "week":
            key = ((received_at_local.date() - start_date).days // 7,)
        elif granularity == "month":
            key = (received_at_local.year, received_at_local.month)
        else:
            key = received_at_local.year

        if key in counts:
            counts[key]["outgoing" if "SENT" in label_ids else "incoming"] += 1

    points = []
    for key, values in counts.items():
        if granularity == "hour":
            label = f"{key[1]:02d}:00" if (end_date - start_date).days <= 1 else f"{key[0].strftime('%b %d')} {key[1]:02d}:00"
        elif granularity == "day":
            label = key[0].strftime("%b %d")
        elif granularity == "week":
            label = f"Week {key[0] + 1}"
        elif granularity == "month":
            label = datetime(key[0], key[1], 1).strftime("%b")
        else:
            label = str(key)
        points.append({"label": label, **values})
    return points

db = SessionLocal()
acc = db.query(GmailAccount).filter(GmailAccount.email == 'marcolie530@gmail.com').first()
client_tz = _resolve_client_timezone("Asia/Kolkata")
now_local = datetime.now(client_tz)

for rk in ["yesterday", "this_week"]:
    s_utc, e_utc, cs_utc, ce_utc, sd, ed = _period_bounds(rk, now_local, client_tz)
    gran = _get_granularity(rk)
    rows = db.execute(
        select(Email.id, Email.received_at, Label.label_id)
        .join(EmailLabel, EmailLabel.email_id == Email.id, isouter=True)
        .join(Label, Label.id == EmailLabel.label_id, isouter=True)
        .where(Email.gmail_account_id == acc.id, Email.received_at >= s_utc, Email.received_at < e_utc)
    ).all()
    pts = _activity_points(rows, sd, ed, gran, client_tz)
    total_pts_inc = sum(p['incoming'] for p in pts)
    print(f"Range '{rk}': {len(pts)} points, total incoming={total_pts_inc}")
    for p in pts:
        if p['incoming'] > 0 or p['outgoing'] > 0:
            print(f"   {p['label']} -> incoming={p['incoming']}, outgoing={p['outgoing']}")

db.close()

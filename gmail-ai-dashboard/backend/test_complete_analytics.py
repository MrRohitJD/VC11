from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
from calendar import monthrange
from typing import Optional

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
    elif range_key == "this_year":
        start_date = today.replace(month=1, day=1)
        end_date = today + timedelta(days=1)
    elif range_key == "last_year":
        start_date = today.replace(year=today.year - 1, month=1, day=1)
        end_date = today.replace(month=1, day=1)
    elif range_key == "all_time":
        start_date = minimum_date or today
        end_date = today + timedelta(days=1)
    elif range_key == "last_7_days":
        start_date = today - timedelta(days=7)
        end_date = today + timedelta(days=1)
    else:
        raise ValueError(f"Unsupported range: {range_key}")

    period_length = end_date - start_date
    comp_start_date = start_date - period_length
    comp_end_date = start_date

    # Exact local boundaries converted to UTC datetimes
    start_dt_local = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=client_tz)
    end_dt_local = datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0, tzinfo=client_tz)
    start_at_utc = start_dt_local.astimezone(timezone.utc)
    end_at_utc = end_dt_local.astimezone(timezone.utc)

    comp_start_dt = datetime(comp_start_date.year, comp_start_date.month, comp_start_date.day, 0, 0, 0, tzinfo=client_tz)
    comp_end_dt = datetime(comp_end_date.year, comp_end_date.month, comp_end_date.day, 0, 0, 0, tzinfo=client_tz)
    comp_start_at_utc = comp_start_dt.astimezone(timezone.utc)
    comp_end_at_utc = comp_end_dt.astimezone(timezone.utc)

    return start_at_utc, end_at_utc, comp_start_at_utc, comp_end_at_utc, start_date, end_date

# Verify
client_tz = _resolve_client_timezone("Asia/Kolkata")
now_local = datetime.now(client_tz)
print("Resolved client timezone:", client_tz)
print("Now local:", now_local)

for rk in ["today", "yesterday", "this_week", "last_week", "this_month", "last_month"]:
    s_utc, e_utc, cs_utc, ce_utc, sd, ed = _period_bounds(rk, now_local, client_tz)
    print(f"[{rk}] Local dates: {sd} -> {ed} | UTC range: {s_utc} -> {e_utc}")

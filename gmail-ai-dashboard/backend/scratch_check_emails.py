import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datetime import datetime, timezone, timedelta
from app.database.connection import SessionLocal
from app.models.email import Email, GmailAccount, User
from collections import Counter
from app.api.routes.emails import _period_bounds, _count_metrics, RANGE_OPTIONS

db = SessionLocal()
account = db.query(GmailAccount).first()
print(f"Gmail Account: {account.id if account else None}, email: {account.email if account else None}")

now_utc = datetime.now(timezone.utc)
now_local = datetime.now()
print(f"Server now_utc: {now_utc} (date: {now_utc.date()})")
print(f"Server now_local: {now_local} (date: {now_local.date()})")

# Check distribution of received_at dates in DB
dates = []
for row in db.query(Email.received_at).filter(Email.gmail_account_id == account.id).all():
    if row[0]:
        dates.append(row[0])

print(f"Total emails for account: {len(dates)}")
if dates:
    print(f"Min date: {min(dates)}")
    print(f"Max date: {max(dates)}")

print("\n--- Emails by date (raw stored received_at.date()): ---")
c_raw = Counter(d.date() for d in dates)
for d, cnt in sorted(c_raw.items(), reverse=True)[:10]:
    print(f"  {d}: {cnt} emails")

print("\n--- Emails in last 5 days converted to UTC date vs Local date (+05:30): ---")
ist = timezone(timedelta(hours=5, minutes=30))
c_utc = Counter(d.astimezone(timezone.utc).date() for d in dates)
c_ist = Counter(d.astimezone(ist).date() for d in dates)
for d in [now_local.date(), now_local.date() - timedelta(days=1), now_local.date() - timedelta(days=2)]:
    print(f"  Date {d} -> in IST: {c_ist.get(d, 0)} emails | in UTC: {c_utc.get(d, 0)} emails")

print("\n--- Current Backend _period_bounds Output (using today = now_utc.date()): ---")
today_utc = now_utc.date()
for rk in ["today", "yesterday", "this_week", "last_week", "this_month", "last_month"]:
    s, e, cs, ce = _period_bounds(rk, today_utc)
    metrics = _count_metrics(db, account.id, s, e)
    print(f"  Range '{rk}': start={s}, end={e} => metrics={metrics}")

print("\n--- Backend _period_bounds Output IF using today = now_local.date() or user local date: ---")
today_local = now_local.date()
for rk in ["today", "yesterday", "this_week", "last_week", "this_month", "last_month"]:
    s, e, cs, ce = _period_bounds(rk, today_local)
    # also note: start_at / end_at timezone!
    start_at_utc = datetime.combine(s, datetime.min.time(), tzinfo=timezone.utc)
    end_at_utc = datetime.combine(e, datetime.min.time(), tzinfo=timezone.utc)
    # vs start_at in IST
    start_at_ist = datetime.combine(s, datetime.min.time(), tzinfo=ist)
    end_at_ist = datetime.combine(e, datetime.min.time(), tzinfo=ist)
    cnt_utc = db.query(Email).filter(Email.gmail_account_id == account.id, Email.received_at >= start_at_utc, Email.received_at < end_at_utc).count()
    cnt_ist = db.query(Email).filter(Email.gmail_account_id == account.id, Email.received_at >= start_at_ist, Email.received_at < end_at_ist).count()
    print(f"  Range '{rk}' (local date {s} to {e}): count_with_utc_bounds={cnt_utc}, count_with_ist_bounds={cnt_ist}")

db.close()

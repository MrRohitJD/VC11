import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datetime import datetime, timezone, timedelta
from app.database.connection import SessionLocal
from app.models.email import Email, GmailAccount

db = SessionLocal()
account = db.query(GmailAccount).first()

emails_oct12 = db.query(Email).filter(
    Email.gmail_account_id == account.id,
    Email.received_at >= datetime(2026, 10, 11, 0, 0, 0, tzinfo=timezone.utc)
).order_by(Email.received_at.desc()).all()

print(f"Total emails from 2026-10-11 UTC onwards: {len(emails_oct12)}")
for e in emails_oct12:
    print(f"ID: {e.id} | received_at: {e.received_at} | UTC: {e.received_at.astimezone(timezone.utc)} | Subject: {e.subject}")

db.close()

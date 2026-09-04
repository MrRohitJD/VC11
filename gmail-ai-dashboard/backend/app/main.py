from app.database.connection import SessionLocal
from app.services.get_mail import get_gmail_service
from app.services.gmail_service import (
    fetch_and_store_emails,
    get_or_create_gmail_account,
)


def import_emails(
    count=20,
    start_date="2026-01-01",
    end_date="2026-01-31",
):
    """Fetch and store emails; return clean records for the caller."""
    service = get_gmail_service()
    db = SessionLocal()
    try:
        account = get_or_create_gmail_account(db, service)
        emails = fetch_and_store_emails(
            db=db,
            gmail_account_id=account.id,
            count=count,
            start_date=start_date,
            end_date=end_date,
            service=service,
        )
        for email in emails:
            print(f"Labels: {email.get('labelIds', [])}")
            print(f"Subject: {email.get('subject', '')}")
        return emails
    finally:
        db.close()
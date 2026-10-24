"""Safe mailbox scope shared by all mailbox assistant retrieval paths."""

from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.models.email import Email, EmailLabel, GmailAccount, Label
from app.rag.query_constraints import billing_email_matches, is_financial_query, received_date_bounds
from app.rag.preprocessing import clean_email_text
from app.rag.search_plan import requested_sender, sender_matches


EXCLUDED_MAILBOX_LABELS = ("SPAM", "TRASH", "DRAFT")
NON_PRIMARY_CATEGORY_LABELS = (
    "CATEGORY_SOCIAL", "CATEGORY_PROMOTIONS", "CATEGORY_UPDATES", "CATEGORY_FORUMS",
)


def mailbox_email_condition():
    """Include every Gmail category while excluding junk and incomplete mail."""
    def has_label(*label_ids):
        return Email.labels.any(EmailLabel.label.has(Label.label_id.in_(label_ids)))

    return ~has_label(*EXCLUDED_MAILBOX_LABELS)


def primary_email_condition():
    """Return the explicit Primary-category condition for label filters."""
    def has_label(*label_ids):
        return Email.labels.any(EmailLabel.label.has(Label.label_id.in_(label_ids)))

    return and_(
        or_(has_label("CATEGORY_PERSONAL"), has_label("INBOX")),
        ~has_label(*NON_PRIMARY_CATEGORY_LABELS, *EXCLUDED_MAILBOX_LABELS),
    )


def _email_ids(db: Session, user_id: UUID | str, query: str, condition) -> list[str]:
    """Read current relational labels, including for already indexed chunks."""
    if db is None:
        raise ValueError("A database session is required for mailbox retrieval")
    scope = [GmailAccount.user_id == UUID(str(user_id)), condition()]
    bounds = received_date_bounds(query)
    if bounds:
        scope.extend([Email.received_at >= bounds[0], Email.received_at < bounds[1]])
    sender = requested_sender(query)
    if sender:
        sender_ids = [e.id for e in db.scalars(select(Email).join(GmailAccount, GmailAccount.id == Email.gmail_account_id).where(*scope))
                      if sender_matches(sender, e.from_email, e.sender_name)]
        scope.append(Email.id.in_(sender_ids))
    if is_financial_query(query):
        return [str(email.id) for email in db.scalars(
            select(Email)
            .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
            .where(*scope)
        ) if billing_email_matches(query, email.subject,
                                  clean_email_text(email.body, email.body_html, remove_quotes=False, remove_signatures=False) or email.snippet,
                                  email.received_at)]
    return [str(email_id) for email_id in db.scalars(
        select(Email.id)
        .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
        .where(*scope)
    )]


def mailbox_email_ids(db: Session, user_id: UUID | str, query: str = "") -> list[str]:
    """Return all non-junk mailbox email IDs, across every Gmail category."""
    return _email_ids(db, user_id, query, mailbox_email_condition)


def primary_email_ids(db: Session, user_id: UUID | str, query: str = "") -> list[str]:
    """Compatibility helper for callers explicitly requesting Primary mail."""
    return _email_ids(db, user_id, query, primary_email_condition)


def restrict_to_email_ids(custom_filter: dict | None, email_ids: list[str]) -> dict:
    """Intersect the scope with existing filters before vector ranking."""
    scope = {"email_id": {"$in": email_ids}}
    if not custom_filter:
        return scope
    # Keep the filter shape compatible with the vector-manager contract.
    conditions = [{key: value} for key, value in custom_filter.items()]
    return {"$and": [*conditions, scope]}

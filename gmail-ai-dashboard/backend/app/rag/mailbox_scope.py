"""Primary-category scope shared by all mailbox assistant retrieval paths."""

from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.models.email import Email, EmailLabel, GmailAccount, Label
from app.rag.query_constraints import billing_email_matches, is_financial_query, received_date_bounds
from app.rag.preprocessing import clean_email_text


NON_PRIMARY_LABELS = (
    "CATEGORY_SOCIAL", "CATEGORY_PROMOTIONS", "CATEGORY_UPDATES", "CATEGORY_FORUMS",
)


def primary_email_condition():
    """Use Gmail system label IDs, never AI categories or custom label names.

    Primary messages can carry CATEGORY_PERSONAL or just INBOX without another
    category. Require positive label evidence so unsynced/unlabelled records
    are not assumed to be Primary. Exclude conflicting categories and junk.
    """
    def has_label(*label_ids):
        return Email.labels.any(EmailLabel.label.has(Label.label_id.in_(label_ids)))

    return and_(
        or_(has_label("CATEGORY_PERSONAL"), has_label("INBOX")),
        ~has_label(*NON_PRIMARY_LABELS, "SPAM", "TRASH", "DRAFT"),
    )


def primary_email_ids(db: Session, user_id: UUID | str, query: str = "") -> list[str]:
    """Read current relational labels, including for already indexed chunks."""
    if db is None:
        raise ValueError("A database session is required for Primary-only retrieval")
    scope = [GmailAccount.user_id == UUID(str(user_id)), primary_email_condition()]
    bounds = received_date_bounds(query)
    if bounds:
        scope.extend([Email.received_at >= bounds[0], Email.received_at < bounds[1]])
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


def restrict_to_email_ids(custom_filter: dict | None, email_ids: list[str]) -> dict:
    """Intersect the scope with existing filters before vector ranking."""
    scope = {"email_id": {"$in": email_ids}}
    if not custom_filter:
        return scope
    # Chroma requires one operator per filter expression.
    conditions = [{key: value} for key, value in custom_filter.items()]
    return {"$and": [*conditions, scope]}

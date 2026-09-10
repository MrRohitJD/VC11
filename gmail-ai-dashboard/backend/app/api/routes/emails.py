"""Email & label query routes — all endpoints enforce user-level authorization."""

import json
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.connection import get_db
from app.models.email import Email, EmailClassification, EmailLabel, GmailAccount, Label, User
from app.services.gmail_service import _parse_message, build_gmail_service_for_user

router = APIRouter(prefix="/api", tags=["emails"])

RANGE_OPTIONS = {
	"today": "Today",
	"this_week": "This week",
	"last_week": "Last week",
	"this_month": "This month",
	"last_month": "Last month",
	"six_months": "6 months",
	"one_year": "1 year",
}

CHART_RANGE_OPTIONS = {
	"today": "Today",
	"last_7_days": "Last 7 days",
	"this_month": "This month",
	"last_month": "Last month",
	"six_months": "6 months",
	"one_year": "1 year",
	"all_time": "All time",
}

SYSTEM_CATEGORY_LABELS = {
	"INBOX", "SENT", "DRAFT", "SPAM", "TRASH", "ALL_MAIL", "IMPORTANT",
	"STARRED", "UNREAD",
}

GMAIL_CATEGORY_NAMES = {
	"CATEGORY_PERSONAL": "Personal",
	"CATEGORY_SOCIAL": "Social",
	"CATEGORY_PROMOTIONS": "Promotions",
	"CATEGORY_UPDATES": "Updates",
	"CATEGORY_FORUMS": "Forums",
}


def _subtract_months(value: date, months: int) -> date:
	target_month = value.month - months
	year = value.year + (target_month - 1) // 12
	month = (target_month - 1) % 12 + 1
	return value.replace(year=year, month=month, day=min(value.day, monthrange(year, month)[1]))


def _period_bounds(range_key: str, today: date) -> tuple[date, date, date, date]:
	if range_key == "today":
		start, end = today, today + timedelta(days=1)
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
		start, end = _subtract_months(today, 6), today + timedelta(days=1)
	elif range_key == "one_year":
		start, end = _subtract_months(today, 12), today + timedelta(days=1)
	else:
		raise HTTPException(status_code=400, detail=f"Unsupported range: {range_key}")

	comparison_start = _subtract_months(start, 1)
	comparison_end = _subtract_months(end, 1)
	return start, end, comparison_start, comparison_end


def _count_metrics(db: Session, account_id, start: date, end: date) -> dict:
	start_at = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
	end_at = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)
	base_filter = (Email.gmail_account_id == account_id, Email.received_at >= start_at, Email.received_at < end_at)
	total = db.scalar(select(func.count(Email.id)).where(*base_filter)) or 0
	unread = db.scalar(select(func.count(Email.id)).where(*base_filter, Email.is_read.is_(False))) or 0
	auto_labeled = db.scalar(
		select(func.count(func.distinct(Email.id)))
		.join(EmailClassification, EmailClassification.email_id == Email.id)
		.where(*base_filter)
	) or 0
	important = db.scalar(
		select(func.count(func.distinct(Email.id)))
		.join(EmailLabel, EmailLabel.email_id == Email.id, isouter=True)
		.join(Label, Label.id == EmailLabel.label_id, isouter=True)
		.join(EmailClassification, EmailClassification.email_id == Email.id, isouter=True)
		.where(
			*base_filter,
			(Label.label_id == "IMPORTANT")
			| func.lower(EmailClassification.label).in_(["important", "high_priority"]),
		)
	) or 0
	return {
		"total": total,
		"auto_labeled": auto_labeled,
		"important": important,
		"unread": unread,
	}


def _activity_period(range_key: str, today: date, minimum_date: date | None) -> tuple[date, date, str]:
	if range_key == "today":
		return today - timedelta(days=1), today + timedelta(days=1), "hour"
	if range_key == "last_7_days":
		return today - timedelta(days=6), today + timedelta(days=1), "day"
	if range_key == "this_month":
		return today.replace(day=1), today + timedelta(days=1), "day"
	if range_key == "last_month":
		end = today.replace(day=1)
		return (end - timedelta(days=1)).replace(day=1), end, "week"
	if range_key == "six_months":
		return _subtract_months(today, 5).replace(day=1), today + timedelta(days=1), "month"
	if range_key == "one_year":
		return _subtract_months(today, 11).replace(day=1), today + timedelta(days=1), "year"
	if range_key == "all_time":
		start = minimum_date or today
		return start.replace(month=1, day=1), today + timedelta(days=1), "year"
	raise HTTPException(status_code=400, detail=f"Unsupported chart range: {range_key}")


def _activity_bucket(value: datetime, granularity: str) -> tuple:
	if granularity == "hour":
		return value.date(), value.hour
	if granularity == "day":
		return value.date()
	if granularity == "week":
		return value.date()
	if granularity == "month":
		return value.year, value.month
	return value.year


def _activity_points(rows: list[tuple], start: date, end: date, granularity: str) -> list[dict]:
	counts: dict[tuple, dict[str, int]] = {}
	if granularity == "hour":
		cursor = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
		end_at = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)
		while cursor < end_at:
			key = (cursor.date(), cursor.hour)
			counts[key] = {"incoming": 0, "outgoing": 0}
			cursor += timedelta(hours=1)
	elif granularity == "day":
		cursor = start
		while cursor < end:
			counts[(cursor,)] = {"incoming": 0, "outgoing": 0}
			cursor += timedelta(days=1)
	elif granularity == "week":
		cursor = start
		while cursor < end:
			counts[(cursor,)] = {"incoming": 0, "outgoing": 0}
			cursor += timedelta(days=7)
	elif granularity == "month":
		cursor = start.replace(day=1)
		while cursor < end:
			counts[(cursor.year, cursor.month)] = {"incoming": 0, "outgoing": 0}
			cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
	else:
		for year in range(start.year, end.year + 1):
			counts[year] = {"incoming": 0, "outgoing": 0}

	for received_at, label_ids in rows:
		if granularity == "week":
			key = ((received_at.date() - start).days // 7,)
		elif granularity == "year":
			key = received_at.year
		else:
			key = _activity_bucket(received_at, granularity)
		if granularity == "day":
			key = (key,)
		if key not in counts:
			continue
		counts[key]["outgoing" if "SENT" in label_ids else "incoming"] += 1

	points = []
	for key, values in counts.items():
		if granularity == "hour":
			label = f"{key[0].strftime('%b %d')} {key[1]:02d}:00"
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


@router.get("/analytics/charts")
def analytics_charts(
	range_key: str = Query("last_7_days", alias="range"),
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""Return live category distribution and incoming/outgoing activity buckets."""
	if range_key not in CHART_RANGE_OPTIONS:
		raise HTTPException(status_code=400, detail=f"Unsupported chart range: {range_key}")
	account = _get_account(db, user)
	today = datetime.now(timezone.utc).date()
	minimum_date = db.scalar(
		select(func.min(Email.received_at)).where(Email.gmail_account_id == account.id)
	)
	minimum_date = minimum_date.date() if minimum_date else None
	start, end, granularity = _activity_period(range_key, today, minimum_date)
	start_at = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
	end_at = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)

	rows = db.execute(
		select(Email.received_at, Label.label_id)
		.join(EmailLabel, EmailLabel.email_id == Email.id, isouter=True)
		.join(Label, Label.id == EmailLabel.label_id, isouter=True)
		.where(
			Email.gmail_account_id == account.id,
			Email.received_at >= start_at,
			Email.received_at < end_at,
		)
	).all()
	activity_rows: dict[datetime, set[str]] = {}
	for received_at, label_id in rows:
		if received_at is not None:
			activity_rows.setdefault(received_at, set()).add(label_id or "")

	category_counts: dict[str, int] = {}
	category_rows = db.execute(
		select(Email.id, Label.label_id, Label.name)
		.join(EmailLabel, EmailLabel.email_id == Email.id, isouter=True)
		.join(Label, Label.id == EmailLabel.label_id, isouter=True)
		.where(Email.gmail_account_id == account.id)
	).all()
	email_categories: dict[object, set[str]] = {}
	for email_id, label_id, name in category_rows:
		if label_id in SYSTEM_CATEGORY_LABELS:
			continue
		category_name = GMAIL_CATEGORY_NAMES.get(label_id, name)
		if category_name:
			email_categories.setdefault(email_id, set()).add(category_name)
	for categories_for_email in email_categories.values():
		for category_name in categories_for_email:
			category_counts[category_name] = category_counts.get(category_name, 0) + 1
	for email_id in {email_id for email_id, _, _ in category_rows} - email_categories.keys():
		category_counts["Uncategorized"] = category_counts.get("Uncategorized", 0) + 1
	top_categories = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)
	category_total = sum(category_counts.values())
	categories = [
		{
			"name": name,
			"count": count,
			"value": round(count / category_total * 100, 1) if category_total else 0,
		}
		for name, count in top_categories[:6]
	]
	other_categories = [
		{"name": name, "count": count}
		for name, count in top_categories[6:]
	]
	other_count = sum(item["count"] for item in other_categories)
	if other_count:
		categories.append({
			"name": "Others",
			"count": other_count,
			"value": round(other_count / category_total * 100, 1),
			"breakdown": other_categories,
		})

	return {
		"range": range_key,
		"label": CHART_RANGE_OPTIONS[range_key],
		"total_emails": db.scalar(
			select(func.count(Email.id)).where(Email.gmail_account_id == account.id)
		) or 0,
		"categories": categories,
		"activity": _activity_points(
			[(received_at, labels) for received_at, labels in activity_rows.items()],
			start, end, granularity,
		),
	}


@router.get("/analytics/summary")
def analytics_summary(
	range_key: str = Query("this_week", alias="range"),
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""Return mailbox metrics for a selected period and the matching prior month."""
	account = _get_account(db, user)
	today = datetime.now(timezone.utc).date()
	start, end, comparison_start, comparison_end = _period_bounds(range_key, today)
	return {
		"range": range_key,
		"label": RANGE_OPTIONS[range_key],
		"start": start.isoformat(),
		"end": end.isoformat(),
		"current": _count_metrics(db, account.id, start, end),
		"comparison": _count_metrics(db, account.id, comparison_start, comparison_end),
	}


def _get_account(db: Session, user: User) -> GmailAccount:
	account = db.scalar(
		select(GmailAccount).where(GmailAccount.user_id == user.id)
	)
	if account is None:
		raise HTTPException(status_code=404, detail="No Gmail account linked")
	return account


@router.get("/emails")
def list_emails(
	page: int = Query(1, ge=1),
	per_page: int = Query(50, ge=1, le=200),
	search: Optional[str] = Query(None),
	label: Optional[str] = Query(None),
	inbox_only: bool = Query(False),
	unread_only: bool = Query(False),
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""List emails for the authenticated user with pagination, search and label filtering."""
	account = _get_account(db, user)

	query = select(Email).where(Email.gmail_account_id == account.id)
	if inbox_only:
		query = query.join(EmailLabel).join(Label).where(Label.label_id == "INBOX")
	if unread_only:
		query = query.where(Email.is_read.is_(False))

	if search:
		pattern = f"%{search}%"
		query = query.where(
			Email.subject.ilike(pattern)
			| Email.from_email.ilike(pattern)
			| Email.snippet.ilike(pattern)
		)

	if label:
		query = query.join(EmailLabel).join(Label).where(Label.name.ilike(label))

	# Total count
	count_query = select(func.count()).select_from(query.subquery())
	total = db.scalar(count_query) or 0

	# Paginated results
	emails = db.scalars(
		query.order_by(Email.received_at.desc().nullslast())
		.offset((page - 1) * per_page)
		.limit(per_page)
	).all()

	return {
		"total": total,
		"page": page,
		"per_page": per_page,
		"emails": [_serialize_email(e) for e in emails],
	}


@router.get("/emails/{email_id}")
def get_email(
	email_id: str,
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""Get a single email by its database UUID. Enforces user ownership."""
	account = _get_account(db, user)
	email = db.scalar(
		select(Email).where(
			Email.id == email_id,
			Email.gmail_account_id == account.id,
		)
	)
	if email is None:
		raise HTTPException(status_code=404, detail="Email not found")
	if not email.body_html:
		try:
			service = build_gmail_service_for_user(db, user)
			raw_message = service.users().messages().get(
				userId="me", id=email.gmail_message_id, format="full"
			).execute()
			parsed = _parse_message(raw_message)
			email.body = parsed.get("body") or email.body
			email.body_html = parsed.get("body_html") or None
			db.commit()
		except Exception:
			db.rollback()
	return _serialize_email(email, include_body=True)


@router.get("/labels")
def list_labels(
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""List all Gmail labels for the authenticated user."""
	account = _get_account(db, user)
	labels = db.scalars(
		select(Label).where(Label.gmail_account_id == account.id).order_by(Label.name)
	).all()
	return {
		"labels": [
			{
				"id": str(l.id),
				"label_id": l.label_id,
				"name": l.name,
				"type": l.type,
			}
			for l in labels
		]
	}


def _serialize_email(email: Email, include_body: bool = False) -> dict:
	result = {
		"id": str(email.id),
		"gmail_message_id": email.gmail_message_id,
		"thread_id": email.thread_id,
		"subject": email.subject,
		"snippet": email.snippet,
		"from_email": email.from_email,
		"to_email": email.to_email,
		"cc": email.cc,
		"bcc": email.bcc,
		"received_at": email.received_at.isoformat() if email.received_at else None,
		"is_read": email.is_read,
		"has_attachment": email.has_attachment,
		"size_estimate": email.size_estimate,
		"labels": [
			{"id": str(el.label.id), "name": el.label.name}
			for el in email.labels
			if el.label
		],
	}
	if include_body:
		result["body"] = email.body
		result["body_html"] = email.body_html
		result["attachments_meta"] = (
			json.loads(email.attachments_meta) if email.attachments_meta else []
		)
	return result

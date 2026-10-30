"""Email & label query routes — all endpoints enforce user-level authorization."""

import json
import logging
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import case, exists, func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.core.security import get_current_user
from app.database.connection import get_db
from app.models.email import AiCategory, Email, EmailClassification, EmailLabel, GmailAccount, Label, User
from app.services.gmail_service import _parse_message, build_gmail_service_for_user, normalize_recipient_header, send_gmail_message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["emails"])

RANGE_OPTIONS = {
	"today": "Today",
	"yesterday": "Yesterday",
	"this_week": "This week",
	"last_week": "Last week",
	"this_month": "This month",
	"last_month": "Last month",
	"six_months": "6 months",
	"one_year": "1 year",
	"this_year": "This year",
	"last_year": "Last year",
	"all_time": "All time",
	"last_7_days": "Last 7 days",
}

CHART_RANGE_OPTIONS = RANGE_OPTIONS

SYSTEM_CATEGORY_LABELS = {
	"INBOX", "SENT", "DRAFT", "SPAM", "TRASH", "ALL_MAIL", "IMPORTANT",
	"STARRED", "UNREAD",
}

GMAIL_CATEGORY_NAMES = {
	"CATEGORY_PERSONAL": "Primary",
	"CATEGORY_SOCIAL": "Social",
	"CATEGORY_PROMOTIONS": "Promotions",
	"CATEGORY_UPDATES": "Updates",
	"CATEGORY_FORUMS": "Forums",
}


def _resolve_client_timezone(tz_name: Optional[str] = None, x_timezone: Optional[str] = None) -> timezone:
	"""Resolve client timezone from query param, X-Timezone header, system host, or fallback."""
	for candidate in (tz_name, x_timezone):
		if candidate:
			try:
				return ZoneInfo(candidate.strip())
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
) -> tuple[datetime, datetime, datetime, datetime, date, date]:
	"""
	Calculates exact date bounds in the client's local timezone, and converts them to
	UTC datetimes for accurate database filtering.
	Returns:
		(start_at_utc, end_at_utc, comp_start_at_utc, comp_end_at_utc, start_date_local, end_date_local)
	"""
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
		return None, None, None, None, start_date, end_date
	elif range_key == "last_7_days":
		start_date = today - timedelta(days=7)
		end_date = today + timedelta(days=1)
	else:
		raise HTTPException(status_code=400, detail=f"Unsupported range: {range_key}")

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


def _count_metrics(db: Session, account_id, start_at: Optional[datetime], end_at: Optional[datetime]) -> dict:
	conditions = [Email.gmail_account_id == account_id]
	if start_at is not None:
		conditions.append(Email.received_at >= start_at)
	if end_at is not None:
		conditions.append(Email.received_at < end_at)

	# Keep this as one row per email and use correlated EXISTS checks. The old
	# implementation ran four separate COUNT queries and joined classifications
	# and labels for the important metric, which multiplied rows and required
	# DISTINCT. One aggregate reduces the database round trips from four to one
	# per period and lets the dashboard classification index answer the EXISTS
	# lookups directly.
	current_classification = exists(
		select(1).where(
			EmailClassification.email_id == Email.id,
			EmailClassification.is_current.is_(True),
			EmailClassification.classification_status == "SUCCEEDED",
		)
	)
	important_label = exists(
		select(1)
		.select_from(EmailLabel)
		.join(Label, Label.id == EmailLabel.label_id)
		.where(
			EmailLabel.email_id == Email.id,
			EmailLabel.gmail_account_id == account_id,
			Label.gmail_account_id == account_id,
			Label.label_id == "IMPORTANT",
		)
	)
	important_ai_category = exists(
		select(1)
		.select_from(EmailClassification)
		.join(AiCategory, AiCategory.id == EmailClassification.category_id)
		.where(
			EmailClassification.email_id == Email.id,
			EmailClassification.is_current.is_(True),
			EmailClassification.classification_status == "SUCCEEDED",
			func.lower(AiCategory.name).in_(("important", "high_priority")),
		)
	)
	metrics = db.execute(
		select(
			func.count(Email.id),
			func.count(case((Email.is_read.is_(False), 1))),
			func.count(case((current_classification, 1))),
			func.count(case((or_(important_label, important_ai_category), 1))),
		).where(*conditions)
	).one()
	total, unread, auto_labeled, important = (value or 0 for value in metrics)
	return {
		"total": total,
		"auto_labeled": auto_labeled,
		"important": important,
		"unread": unread,
	}


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


@router.get("/analytics/charts")
def analytics_charts(
	range_key: str = Query("this_week", alias="range"),
	tz: Optional[str] = Query(None),
	x_timezone: Optional[str] = Header(None, alias="X-Timezone"),
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""Return live category distribution and incoming/outgoing activity buckets."""
	if range_key not in CHART_RANGE_OPTIONS:
		raise HTTPException(status_code=400, detail=f"Unsupported chart range: {range_key}")
	account = _get_account(db, user)
	client_tz = _resolve_client_timezone(tz, x_timezone)
	now_local = datetime.now(client_tz)
	min_date = None
	if range_key == "all_time":
		minimum_date = db.scalar(
			select(func.min(Email.received_at)).where(Email.gmail_account_id == account.id)
		)
		min_date = minimum_date.astimezone(client_tz).date() if minimum_date else None
	start_at, end_at, _, _, start_date, end_date = _period_bounds(
		range_key, now_local, client_tz, min_date
	)
	granularity = _get_granularity(range_key)

	time_conditions = [Email.gmail_account_id == account.id]
	if start_at is not None:
		time_conditions.append(Email.received_at >= start_at)
	if end_at is not None:
		time_conditions.append(Email.received_at < end_at)

	# This single label query feeds both charts. Previously the same filtered
	# email set was scanned twice: once for activity and once for categories.
	rows = db.execute(
		select(Email.id, Email.received_at, Label.label_id, Label.name)
		.join(EmailLabel, EmailLabel.email_id == Email.id, isouter=True)
		.join(Label, Label.id == EmailLabel.label_id, isouter=True)
		.where(*time_conditions)
	).all()

	category_counts: dict[str, int] = {}
	email_categories: dict[object, set[str]] = {}
	for email_id, _, label_id, name in rows:
		if label_id in SYSTEM_CATEGORY_LABELS:
			continue
		if user.is_demo and label_id not in GMAIL_CATEGORY_NAMES:
			continue
		category_name = GMAIL_CATEGORY_NAMES.get(label_id, name)
		if category_name:
			email_categories.setdefault(email_id, set()).add(category_name)

	# Fetch AI classification categories for these emails
	ai_class_rows = db.execute(
		select(Email.id, AiCategory.name)
		.join(EmailClassification, EmailClassification.email_id == Email.id)
		.join(AiCategory, AiCategory.id == EmailClassification.category_id)
		.where(
			*time_conditions,
			EmailClassification.is_current.is_(True),
			EmailClassification.classification_status == "SUCCEEDED",
		)
	).all()
	ai_categories_by_email: dict[object, str] = {
		eid: cat_name for eid, cat_name in ai_class_rows if cat_name
	}

	all_email_ids = {eid for eid, _, _, _ in rows} | set(ai_categories_by_email.keys())
	for eid in all_email_ids:
		if user.is_demo and eid in email_categories and email_categories[eid]:
			for cat_name in email_categories[eid]:
				category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
		elif eid in ai_categories_by_email:
			cat_name = ai_categories_by_email[eid]
			category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
		elif eid in email_categories and email_categories[eid]:
			for cat_name in email_categories[eid]:
				category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
		else:
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

	# The outer join returns at least one row for every email, so the unique IDs
	# already represent the total and avoid a fourth scan of the time range.
	total_emails = len({eid for eid, _, _, _ in rows})

	return {
		"range": range_key,
		"label": CHART_RANGE_OPTIONS[range_key],
		"total_emails": total_emails,
		"categories": categories,
		"activity": _activity_points(
			[(email_id, received_at, label_id) for email_id, received_at, label_id, _ in rows],
			start_date, end_date, granularity, client_tz
		),
	}


@router.get("/analytics/summary")
def analytics_summary(
	range_key: str = Query("this_week", alias="range"),
	tz: Optional[str] = Query(None),
	x_timezone: Optional[str] = Header(None, alias="X-Timezone"),
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""Return mailbox metrics for a selected period and the matching prior comparison period."""
	account = _get_account(db, user)
	try:
		from app.services.llm_classification_service import ensure_user_emails_classified
		ensure_user_emails_classified(account.id)
	except Exception:
		pass
	client_tz = _resolve_client_timezone(tz, x_timezone)
	now_local = datetime.now(client_tz)
	if range_key not in RANGE_OPTIONS:
		raise HTTPException(status_code=400, detail=f"Unsupported range: {range_key}")
	min_date = None
	if range_key == "all_time":
		minimum_date = db.scalar(
			select(func.min(Email.received_at)).where(Email.gmail_account_id == account.id)
		)
		min_date = minimum_date.astimezone(client_tz).date() if minimum_date else None
	start_at, end_at, comp_start_at, comp_end_at, start_date, end_date = _period_bounds(
		range_key, now_local, client_tz, min_date
	)
	current_metrics = _count_metrics(db, account.id, start_at, end_at)
	comp_metrics = (
		_count_metrics(db, account.id, comp_start_at, comp_end_at)
		if (comp_start_at and comp_end_at)
		else {k: 0 for k in current_metrics}
	)
	return {
		"range": range_key,
		"label": RANGE_OPTIONS[range_key],
		"start": start_date.isoformat(),
		"end": end_date.isoformat(),
		"current": current_metrics,
		"comparison": comp_metrics,
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
	sent_only: bool = Query(False),
	unread_only: bool = Query(False),
	range_key: Optional[str] = Query(None, alias="range"),
	tz: Optional[str] = Query(None),
	x_timezone: Optional[str] = Header(None, alias="X-Timezone"),
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""List emails for the authenticated user with pagination, search and label filtering."""
	account = _get_account(db, user)
	if not user.is_demo:
		try:
			from app.services.llm_classification_service import ensure_user_emails_classified
			ensure_user_emails_classified(account.id)
		except Exception:
			pass
		try:
			from app.services.rag_auto_indexer import ensure_user_emails_embedded
			ensure_user_emails_embedded(user.id)
		except Exception:
			pass
	start_at = None
	end_at = None
	if isinstance(range_key, str):
		if range_key not in RANGE_OPTIONS:
			raise HTTPException(status_code=400, detail=f"Unsupported range: {range_key}")
		client_tz = _resolve_client_timezone(tz, x_timezone)
		now_local = datetime.now(client_tz)
		min_date = None
		if range_key == "all_time":
			minimum_date = db.scalar(
				select(func.min(Email.received_at)).where(Email.gmail_account_id == account.id)
			)
			min_date = minimum_date.astimezone(client_tz).date() if minimum_date else None
		start_at, end_at, _, _, _, _ = _period_bounds(
			range_key, now_local, client_tz, min_date
		)

	query = select(Email).where(Email.gmail_account_id == account.id)
	if isinstance(range_key, str) and start_at and end_at:
		query = query.where(Email.received_at >= start_at, Email.received_at < end_at)
	if inbox_only:
		query = query.join(EmailLabel).join(Label).where(Label.label_id == "INBOX")
	elif sent_only:
		query = query.where(
			Email.labels.any(
				EmailLabel.label.has(
					or_(
						Label.label_id.ilike("SENT"),
						Label.name.ilike("SENT"),
					)
				)
			)
			| (Email.from_email.ilike(f"%{account.email}%"))
		)
	if unread_only:
		query = query.where(Email.is_read.is_(False))

	if search and isinstance(search, str):
		pattern = f"%{search}%"
		query = query.where(
			Email.subject.ilike(pattern)
			| Email.from_email.ilike(pattern)
			| Email.to_email.ilike(pattern)
			| Email.snippet.ilike(pattern)
		)

	if label and isinstance(label, str):
		query = query.where(
			Email.labels.any(
				EmailLabel.label.has(
					or_(
						Label.name.ilike(label),
						Label.label_id.ilike(label),
					)
				)
			)
			| Email.classifications.any(EmailClassification.category.has(AiCategory.name.ilike(label)))
		)

	# Total count
	count_query = select(func.count()).select_from(query.subquery())
	total = db.scalar(count_query) or 0

	# Paginated results
	emails = db.scalars(
		query.options(
			selectinload(Email.labels).selectinload(EmailLabel.label),
			selectinload(Email.classifications).selectinload(EmailClassification.category),
		).order_by(Email.received_at.desc().nullslast())
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
	if not user.is_demo and not email.body_html:
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
	ai_category_name = None
	gmail_category = next((GMAIL_CATEGORY_NAMES[link.label.label_id]
	                       for link in (email.labels or [])
	                       if link.label and link.label.label_id in GMAIL_CATEGORY_NAMES), None)
	# First check classifications relationship
	if email.classifications:
		for c in email.classifications:
			if c.is_current and c.classification_status == "SUCCEEDED" and c.category:
				ai_category_name = c.category.name
				break

	# Fallback to checking label of type "ai"
	if not ai_category_name and email.labels:
		for el in email.labels:
			if el.label and (el.label.type == "ai" or el.label.label_id.startswith("AI_")):
				ai_category_name = el.label.name
				break

	labels_list = []
	if ai_category_name:
		labels_list.append({
			"id": f"ai-{ai_category_name.lower()}",
			"name": ai_category_name,
			"is_ai": True,
		})

	for el in (email.labels or []):
		if el.label and el.label.name != ai_category_name:
			labels_list.append({
				"id": str(el.label.id),
				"name": el.label.name,
				"is_ai": el.label.type == "ai" or el.label.label_id.startswith("AI_"),
			})

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
		"ai_label": ai_category_name,
		"gmail_category": gmail_category,
		"labels": labels_list,
	}
	if include_body:
		result["body"] = email.body
		result["body_html"] = email.body_html
		result["attachments_meta"] = (
			email.attachments_meta or []
		)
	return result


class SendEmailRequest(BaseModel):
	to: Optional[str] = None
	recipient: Optional[str] = None
	recipient_email: Optional[str] = None
	subject: Optional[str] = ""
	body: Optional[str] = None
	draft_body: Optional[str] = None
	message: Optional[str] = None
	cc: Optional[str] = None
	bcc: Optional[str] = None

	@property
	def resolved_recipient(self) -> str:
		val = (self.to or self.recipient or self.recipient_email or "").strip()
		if not val:
			raise ValueError("A valid recipient email address is required")
		return val

	@property
	def resolved_body(self) -> str:
		return (self.body or self.draft_body or self.message or "").strip()


@router.post("/emails/send")
@router.post("/gmail/send")
def send_email_endpoint(
	payload: SendEmailRequest,
	user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""Send an email through the user's Gmail account (or simulated in demo mode)."""
	try:
		raw_recipient = payload.resolved_recipient
	except ValueError as exc:
		raise HTTPException(status_code=422, detail=str(exc))

	try:
		recipient = normalize_recipient_header(raw_recipient, "recipient", required=True)
	except Exception as exc:
		raise HTTPException(status_code=422, detail=str(exc))

	cc = None
	if payload.cc:
		try:
			cc = normalize_recipient_header(payload.cc, "Cc")
		except Exception as exc:
			raise HTTPException(status_code=422, detail=str(exc))

	bcc = None
	if payload.bcc:
		try:
			bcc = normalize_recipient_header(payload.bcc, "Bcc")
		except Exception as exc:
			raise HTTPException(status_code=422, detail=str(exc))

	subject = (payload.subject or "").strip()
	body = payload.resolved_body

	account = _get_account(db, user)
	now = datetime.now(timezone.utc)

	if user.is_demo:
		delivery_id = f"demo-sent-{uuid4().hex[:12]}"
		try:
			sent_label = db.scalar(
				select(Label).where(Label.gmail_account_id == account.id, Label.label_id == "SENT")
			)
			if not sent_label:
				sent_label = Label(
					id=uuid4(),
					gmail_account_id=account.id,
					label_id="SENT",
					name="SENT",
					type="SYSTEM",
				)
				db.add(sent_label)
				db.flush()

			sent_email = Email(
				id=uuid4(),
				gmail_account_id=account.id,
				gmail_message_id=delivery_id,
				thread_id=f"thread-{delivery_id}",
				from_email=user.email,
				sender_name=user.name or user.email,
				to_email=recipient,
				cc=cc,
				bcc=bcc,
				subject=subject,
				snippet=body[:150] if body else "",
				body=body,
				body_html=f"<p>{body}</p>" if body else "",
				received_at=now,
				is_read=True,
				is_starred=False,
				has_attachment=False,
			)
			db.add(sent_email)
			db.flush()
			db.add(EmailLabel(email_id=sent_email.id, label_id=sent_label.id, gmail_account_id=account.id))
			db.commit()
		except Exception as exc:
			logger.warning("Could not persist demo sent email: %s", exc)
			db.rollback()

		return {
			"status": "SENT",
			"message": "Email sent successfully (demo mode).",
			"delivery_message_id": delivery_id,
			"demo": True,
		}

	try:
		delivery = send_gmail_message(
			db=db,
			user=user,
			recipient=recipient,
			subject=subject,
			body=body,
			cc=cc,
			bcc=bcc,
		)
		delivery_id = delivery.get("id") if isinstance(delivery, dict) else str(delivery)
	except PermissionError as exc:
		raise HTTPException(status_code=403, detail=str(exc))
	except ValueError as exc:
		raise HTTPException(status_code=422, detail=str(exc))
	except Exception as exc:
		logger.exception("Failed to send email via Gmail API: %s", exc)
		raise HTTPException(status_code=502, detail=f"Failed to deliver email via Gmail: {str(exc)}")

	try:
		sent_label = db.scalar(
			select(Label).where(Label.gmail_account_id == account.id, Label.label_id == "SENT")
		)
		if not sent_label:
			sent_label = Label(
				id=uuid4(),
				gmail_account_id=account.id,
				label_id="SENT",
				name="SENT",
				type="SYSTEM",
			)
			db.add(sent_label)
			db.flush()

		sent_email = Email(
			id=uuid4(),
			gmail_account_id=account.id,
			gmail_message_id=delivery_id or f"sent-{uuid4().hex[:12]}",
			thread_id=delivery.get("threadId") if isinstance(delivery, dict) else None,
			from_email=user.email,
			sender_name=user.name or user.email,
			to_email=recipient,
			cc=cc,
			bcc=bcc,
			subject=subject,
			snippet=body[:150] if body else "",
			body=body,
			body_html=f"<p>{body}</p>" if body else "",
			received_at=now,
			is_read=True,
			is_starred=False,
			has_attachment=False,
		)
		db.add(sent_email)
		db.flush()
		db.add(EmailLabel(email_id=sent_email.id, label_id=sent_label.id, gmail_account_id=account.id))
		db.commit()
	except Exception as exc:
		logger.warning("Could not persist sent email cache: %s", exc)
		db.rollback()

	return {
		"status": "SENT",
		"message": "Email sent successfully.",
		"delivery_message_id": delivery_id,
		"demo": False,
	}

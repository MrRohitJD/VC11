"""Security audit event logging and abuse signal tracking."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from app.models.rate_limit import SecurityEvent

logger = logging.getLogger(__name__)


def record_security_event(
    db: Session,
    ip_hash: str,
    event_type: str,
    severity: str,
    endpoint: str,
    http_method: str,
    status_code: int,
    visitor_hash: Optional[str] = None,
    user_id: Optional[UUID] = None,
    detail: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[SecurityEvent]:
    """Record an audit entry into security_events.

    Designed to never raise or disrupt the calling request if logging fails.
    """
    try:
        event = SecurityEvent(
            id=uuid4(),
            visitor_hash=visitor_hash,
            ip_hash=ip_hash,
            user_id=user_id,
            event_type=event_type,
            severity=severity.upper(),
            endpoint=endpoint[:256],
            http_method=http_method[:10].upper(),
            status_code=status_code,
            detail=detail[:1024] if detail else None,
            metadata_json=json.dumps(metadata) if metadata else None,
            created_at=datetime.now(timezone.utc),
        )
        db.add(event)
        db.commit()
        logger.info(
            "SecurityEvent logged: type=%s severity=%s endpoint=%s ip_hash=%s",
            event_type,
            severity,
            endpoint,
            ip_hash,
        )
        return event
    except Exception as exc:
        logger.warning("Could not record security event: %s", type(exc).__name__)
        try:
            db.rollback()
        except Exception:
            pass
        return None

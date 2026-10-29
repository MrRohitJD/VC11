"""The sole database engine, session factory, and FastAPI dependency.

Schema changes are intentionally not performed here. Apply the versioned
Alembic migrations before starting the API; startup only verifies the schema.
"""

from collections.abc import Generator
import logging

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
import app.models.llm  # noqa: F401 - registers job mapper before sessions are used
import app.models.chat_quota  # noqa: F401 - registers quota and concurrency lease mappers

logger = logging.getLogger(__name__)

_connect_args: dict[str, object] = {}
if settings.database_url.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

_engine_options: dict[str, object] = {"pool_pre_ping": True, "connect_args": _connect_args}
if not settings.database_url.startswith("sqlite"):
    _engine_options.update(
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
    )
engine: Engine = create_engine(settings.database_url, **_engine_options)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


REQUIRED_TABLES = frozenset({
    "users", "gmail_accounts", "emails", "labels", "email_labels",
    "ai_categories", "email_classifications", "sync_history",
    "llm_classification_jobs", "chat_quotas", "chat_concurrency_locks",
    "chat_request_leases", "email_send_approvals",
})

REQUIRED_COLUMNS = {
    "gmail_accounts": frozenset({
        "inbox_sync_page_token", "inbox_sync_heartbeat_at",
    }),
    "email_send_approvals": frozenset({
        "cc", "bcc", "scheduled_for", "delivery_message_id",
    }),
}


def verify_schema() -> None:
    """Fail clearly when the configured database has not been migrated."""
    existing = set(inspect(engine).get_table_names())
    missing = REQUIRED_TABLES - existing
    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Database schema is not current (missing: {names}). "
            "Run `alembic upgrade head` from backend before starting the API."
        )

    missing_columns = {
        table: sorted(columns - {column["name"] for column in inspect(engine).get_columns(table)})
        for table, columns in REQUIRED_COLUMNS.items()
    }
    missing_columns = {table: columns for table, columns in missing_columns.items() if columns}
    if missing_columns:
        details = ", ".join(
            f"{table}: {', '.join(columns)}" for table, columns in missing_columns.items()
        )
        raise RuntimeError(
            f"Database schema is not current (missing columns: {details}). "
            "Run `alembic upgrade head` from backend before starting the API."
        )
    logger.info("Database schema verified")


# Backwards-compatible import for callers that previously invoked init_db.
# It no longer creates or alters tables, avoiding unversioned startup changes.
init_db = verify_schema

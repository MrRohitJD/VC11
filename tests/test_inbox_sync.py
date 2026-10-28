"""The sign-in sync must open quickly and then exhaust Inbox pagination."""

import threading
from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.email import Base, Email, EmailLabel, GmailAccount, Label, User
from app.services import gmail_service


class _Request:
    def __init__(self, response, error=None):
        self.response = response
        self.error = error

    def execute(self):
        if self.error:
            raise self.error
        return self.response


class _Messages:
    def __init__(self, message_ids):
        self.message_ids = message_ids
        self.calls = []
        self.fail_once_at = None

    def list(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_once_at is not None and kwargs.get("pageToken") == self.fail_once_at:
            self.fail_once_at = None
            return _Request(None, RuntimeError("temporary list failure"))
        start = int(kwargs.get("pageToken", "0"))
        end = start + kwargs["maxResults"]
        page = self.message_ids[start:end]
        response = {"messages": [{"id": message_id} for message_id in page]}
        if end < len(self.message_ids):
            response["nextPageToken"] = str(end)
        return _Request(response)


class _Gmail:
    def __init__(self, message_ids):
        self.api = _Messages(message_ids)

    def users(self):
        return self

    def messages(self):
        return self.api


def test_first_100_inbox_messages_then_every_remaining_page(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    message_ids = [f"inbox-{i}" for i in range(1201)]
    service = _Gmail(message_ids)
    fetched_ids = []
    deferred_threads = []
    background_events = []

    class _DeferredThread:
        def __init__(self, *, target, args, kwargs, **_):
            self.target = target
            self.args = args
            self.kwargs = kwargs
            deferred_threads.append(self)

        def start(self):
            pass

        def run(self):
            self.target(*self.args, **self.kwargs)

    def sync_labels(db, account_id, _service):
        label = db.scalar(select(Label).where(Label.gmail_account_id == account_id, Label.label_id == "INBOX"))
        if label is None:
            label = Label(gmail_account_id=account_id, label_id="INBOX", name="INBOX")
            db.add(label)
            db.commit()
        return {"INBOX": label}

    def fetch_messages(_service, ids, *, progress_callback=None):
        fetched_ids.extend(ids)
        if progress_callback:
            progress_callback("fetching", len(ids), len(ids))
        return [{"id": message_id, "labelIds": ["INBOX"], "payload": {"headers": []}} for message_id in ids]

    monkeypatch.setattr(gmail_service, "SessionLocal", sessions)
    monkeypatch.setattr(gmail_service, "build_gmail_service_for_user", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(gmail_service, "sync_all_labels", sync_labels)
    monkeypatch.setattr(gmail_service, "_batch_fetch_messages", fetch_messages)
    monkeypatch.setattr(gmail_service, "_schedule_ai_processing", lambda _account_id: None)
    monkeypatch.setattr(threading, "Thread", _DeferredThread)

    with sessions() as db:
        user = User(google_id="test-user", email="inbox@example.com")
        account = GmailAccount(user=user, email="inbox@example.com", access_token="token")
        db.add_all([user, account])
        db.commit()

        result = gmail_service.sync_all_emails(
            db, user, service,
            background_callback=lambda *event: background_events.append(event),
        )
        assert result["emails_synced"] == 100
        assert result["background_syncing"] is True
        assert len(fetched_ids) == 100
        assert db.scalar(select(func.count()).select_from(Email)) == 100
        assert len(service.api.calls) == 1
        assert deferred_threads

        service.api.fail_once_at = "600"
        deferred_threads.pop().run()

        db.refresh(account)
        assert db.scalar(select(func.count()).select_from(Email)) == 600
        assert account.inbox_sync_page_token == "600"
        assert account.sync_status == "BACKFILL_ERROR"
        assert background_events[-1][0] == "error"

        # The next login resumes at the saved cursor after refreshing the first 100.
        gmail_service.sync_all_emails(
            db, user, service,
            background_callback=lambda *event: background_events.append(event),
        )
        deferred_threads.pop().run()

        assert db.scalar(select(func.count()).select_from(Email)) == 1201
        assert len(fetched_ids) == 1201
        assert [call["maxResults"] for call in service.api.calls] == [100, 500, 500, 100, 500, 500]
        assert all(call["labelIds"] == ["INBOX"] for call in service.api.calls)
        assert background_events[-1][0] == "completed"
        db.refresh(account)
        assert account.sync_status == "IDLE"
        assert account.inbox_sync_page_token is None

        # A later scan can revisit every page without downloading messages twice.
        gmail_service.sync_all_emails(db, user, service)
        deferred_threads.pop().run()
        assert len(fetched_ids) == 1201


def test_sync_reservation_blocks_overlap_and_recovers_stale_worker(monkeypatch):
    from app.api.routes import gmail as gmail_routes

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    started = []

    class _DeferredThread:
        def __init__(self, **kwargs):
            started.append(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(threading, "Thread", _DeferredThread)
    gmail_routes._sync_state.clear()
    try:
        with sessions() as db:
            user = User(google_id="lease-user", email="lease@example.com")
            account = GmailAccount(user=user, email="lease@example.com", access_token="token")
            db.add_all([user, account])
            db.commit()

            first = gmail_routes.start_sync(user=user, db=db)
            assert first["message"] == "Sync started"
            assert len(started) == 1

            # Simulate another web process with no local in-memory state.
            gmail_routes._sync_state.clear()
            second = gmail_routes.start_sync(user=user, db=db)
            assert second["message"] == "Sync already in progress"
            assert len(started) == 1

            account.sync_status = "BACKFILL"
            account.inbox_sync_heartbeat_at = datetime.now(timezone.utc) - timedelta(minutes=16)
            db.commit()
            assert gmail_routes._get_sync_state(str(user.id), db)["status"] == "idle"

            resumed = gmail_routes.start_sync(user=user, db=db)
            assert resumed["message"] == "Sync started"
            assert len(started) == 2
    finally:
        gmail_routes._sync_state.clear()


def test_inbox_cursor_migration_adds_and_removes_columns():
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "gmail-ai-dashboard/backend/alembic/versions/20261028_0013_inbox_sync_cursor.py"
    )
    spec = spec_from_file_location("inbox_cursor_migration", migration_path)
    migration = module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE gmail_accounts (id INTEGER PRIMARY KEY)"))
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()
            columns = {column["name"] for column in inspect(connection).get_columns("gmail_accounts")}
            assert {"inbox_sync_page_token", "inbox_sync_heartbeat_at"} <= columns
            migration.downgrade()
            columns = {column["name"] for column in inspect(connection).get_columns("gmail_accounts")}
            assert "inbox_sync_page_token" not in columns
            assert "inbox_sync_heartbeat_at" not in columns


def test_existing_message_gets_inbox_label_without_download(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(gmail_service, "_batch_fetch_messages", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Existing mail was re-downloaded")))
    with sessions() as db:
        user = User(google_id="old-user", email="old@example.com")
        account = GmailAccount(user=user, email="old@example.com", access_token="token")
        db.add_all([user, account])
        db.commit()
        label = Label(gmail_account_id=account.id, label_id="INBOX", name="INBOX")
        email = Email(gmail_account_id=account.id, gmail_message_id="already-stored")
        db.add_all([label, email])
        db.commit()

        stored, existing = gmail_service._sync_inbox_batch(
            db, account.id, object(), {"INBOX": label}, ["already-stored"]
        )
        assert (stored, existing) == (0, 1)
        assert db.get(EmailLabel, (email.id, label.id)) is not None

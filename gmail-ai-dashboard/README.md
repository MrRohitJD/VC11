# Developer settings: personal database and LLM key

Signed-in users can configure a personal PostgreSQL database and a Google Gemini
or OpenAI API key in Account Settings → Developer settings. Either setting can be
used on its own. Without them, the existing deployment database, provider
fallbacks, and 50-request daily assistant quota remain in effect. A personal
key is used for assistant requests and email classification, with a daily
assistant quota configurable with `DEVELOPER_DAILY_CHAT_LIMIT` (500 by default)
and the same five-request concurrency cap. Embeddings
continue to use the deployment's configured embedding provider.

Set a stable, random `DEVELOPER_SETTINGS_ENCRYPTION_KEY` of at least 32
characters (or a strong `SESSION_SECRET`) on the backend before saving
credentials. Changing that secret makes saved credentials unreadable. API keys
and database URLs are encrypted at rest; the settings API returns only saved
status and a database address without password or query parameters.

Apply `alembic upgrade head` to the deployment database when updating the app.
The personal database must be a reachable PostgreSQL database with this app's
current Alembic schema. Initialize it from `gmail-ai-dashboard/backend` using
`DATABASE_URL=<personal URL> alembic upgrade head` in your own shell, then save
the URL in settings. Connecting only verifies the database and saves its URL.
On first use, the app creates the user's identity and Gmail authorization in
the personal database so future syncs can run there. Existing mailbox data,
labels, classifications, RAG indexes, and assistant history are never copied.
After the switch, new syncs, labels, classifications, searches, exports, and
assistant activity use the personal database. Switching back restores access to
the deployment database's existing data. The
deployment database retains account identity, Gmail authorization, and encrypted
settings so the user can sign in and change connections.

The Developer settings download button generates `gmail-ai-dashboard-schema.sql`
from the current SQLAlchemy model metadata at download time. This keeps the
downloaded schema synchronized with newly added tables and columns; run the
Alembic migrations afterward for production upgrades and indexes that are
managed by migrations.

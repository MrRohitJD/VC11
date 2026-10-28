# Gmail AI Dashboard Backend

## Mailbox assistant category scope

The mailbox assistant answers using only Primary emails. Semantic search,
database fallback, multi-query search, threads, summaries, and attachments all
use the current synced Gmail labels. Other categories are never included as a
fallback when Primary has no matches.

Primary includes `CATEGORY_PERSONAL` and `INBOX` messages without another Gmail
category. Social, Promotions, Updates, Forums, Spam, Trash, and Drafts are
excluded. Records without category or inbox label evidence are excluded until
labels are synced. AI classifications and custom labels named "Primary" do not
determine Gmail category membership. See Google's [system label reference](https://developers.google.com/workspace/gmail/api/guides/labels).

Existing vector indexes work without reindexing: eligible email IDs are read
from the database and applied before vector ranking. Gmail label changes take
effect after sync. This restriction applies to assistant retrieval; mailbox
sync and dashboard email lists continue to include all categories.

Standalone questions do not inherit topics from earlier conversation turns.
For pending bills due this month, retrieval requires explicit billing,
payment-status, and due-date evidence; receipt dates are not treated as payment
deadlines. Calendar boundaries use the browser's `X-Timezone` header. Emails
with unclear status or dates are not reported as confirmed matches.

Citation metadata is refreshed from the authorized email record, including for
older indexes. The assistant's View email action fetches the full message from
`/api/rag/sources/{email_id}`, with sender, recipients, timestamp, plain text,
HTML, and attachments, rather than displaying the compressed citation body.

Relative email date requests such as "interview related emails of this month"
are restricted to the client's calendar month before vector and SQL ranking.
The original date restriction is retained through multi-query expansion.

After retrieval, an LLM receives each authorized candidate's ID, subject,
sender address/name, receipt timestamp, and full message text. It returns an
explicit relevance decision with supporting key points. Only accepted
candidates become answer context and related-email cards. Invalid decisions,
unknown IDs, or provider failures produce no unvalidated sources and a clear
validation-unavailable response. The assistant uses configured Azure OpenAI
first, then NVIDIA, then ExperientialLabs when an earlier provider fails or
returns no usable answer.

"Find all" requests validate the full retrieved candidate set in batches and
retain all accepted sources instead of silently truncating the result to five.

## PostgreSQL and pgvector setup

Retrieval and indexing use the configured PostgreSQL database with pgvector.
Set `RAG_VECTOR_BACKEND=postgres` (the application default) and ensure the
database has the `vector` extension. Local ChromaDB storage is no longer
supported; all vectors and searchable email metadata live in
`email_embeddings`.

Apply the migrations, including the pgvector embedding metadata and HNSW index,
before indexing mail:

```bash
alembic upgrade head
```

The configured database name is `gmail_ai_dashboard`.

1. Create the database as a PostgreSQL role with permission to create databases:

```bash
createdb -U postgres "gmail_ai_dashboard"
```

2. Copy the PostgreSQL connection URL into `backend/.env` and replace the local credentials:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/gmail_ai_dashboard
```

3. Install backend dependencies from `backend`:

```bash
pip install -r requirements.txt
```

4. Apply the schema directly with PostgreSQL:

```bash
psql -U postgres -d "gmail_ai_dashboard" -f app/database/schema.sql
```

The schema script is safe to rerun: it creates missing tables and adds the
email metadata columns to databases created with an older version.

Alternatively, create the tables from the SQLAlchemy models:

```bash
python -c "from app.database.connection import init_db; init_db()"
```

Application startup runs the same idempotent reconciliation automatically.

The database connection is provided by `app.database.connection.get_db`, and the models are defined in `app.models.email`.

## Fetch and store emails from Python

The fetch function returns clean dictionaries and does not print anything. The persistence function stores them and returns the same dictionaries for your own output handling:

```python
from uuid import UUID

from app.database.connection import SessionLocal
from app.services.gmail_service import fetch_and_store_emails

db = SessionLocal()
try:
	result = fetch_and_store_emails(
		db=db,
		gmail_account_id=UUID("YOUR_GMAIL_ACCOUNT_UUID"),
		count=20,
		start_date="2026-01-01",
		end_date="2026-01-31",
	)
finally:
	db.close()

# Print or return result wherever you need it.
```

The `gmail_account_id` must already exist in `gmail_accounts`. Re-running the function updates matching Gmail messages by `gmail_message_id` instead of inserting duplicates.

## Google OAuth

Create a Google OAuth 2.0 **Web application** client and add this authorized redirect URI:

```text
http://127.0.0.1:8000/api/auth/callback
```

Create `backend/.env` with the database, Google client, and session settings:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/gmail_ai_dashboard
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret
BACKEND_URL=http://127.0.0.1:8000
FRONTEND_URL=http://localhost:5173
SESSION_SECRET=replace-with-a-long-random-secret
SECURE_COOKIES=false
```

Start the API from `backend`:

```bash
uvicorn app.api.app:app --reload --port 8000
```

Open `http://localhost:5173/login` or `http://localhost:5173/signup`. Both Google buttons use the same OAuth flow. Google `sub` is stored as `users.google_id`; repeated sign-ins update the existing `users` and `gmail_accounts` records instead of creating duplicates. The API sets an HttpOnly session cookie after the callback.
## Public demo mailbox

The login and signup pages offer a read-only `demo@gmail.com` mailbox. Its
messages, Gmail categories, and AI labels are fictional database records; it
has no Google OAuth token and does not call Gmail. The demo assistant stays
unavailable until embeddings are added in a later step.

Apply the versioned migrations, then seed the mailbox from `backend/`:

```sh
alembic upgrade head
python scripts/seed_demo_mailbox.py
python scripts/seed_demo_mailbox.py --verify
python scripts/verify_demo_api.py
```

The seed is idempotent and does not create chunks or embeddings. When migrating
a database whose schema was cloned from the `20261013_0002` revision but whose
`alembic_version` table is empty, verify the schema first, then stamp that
revision before upgrading:

```sh
python scripts/seed_demo_mailbox.py --check
alembic stamp 20261013_0002
alembic upgrade head
```

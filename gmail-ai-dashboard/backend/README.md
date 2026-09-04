# Gmail AI Dashboard Backend

## PostgreSQL setup

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

Alternatively, create the tables from the SQLAlchemy models:

```bash
python -c "from app.database.connection import init_db; init_db()"
```

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

# VC11 — Complete Backend API Catalogue & Architecture Flow

> **Verification:** Verified directly against `backend/app/api/app.py`, routers (`auth.py`, `emails.py`, `gmail.py`, `llm.py`, `rag.py`, `export.py`, `developer.py`), dependencies (`dependencies.py`, `security.py`), and backend services.

---

## Table of Contents

1. [Backend Overview & Entrypoint Architecture](#1-backend-overview--entrypoint-architecture)
2. [FastAPI Initialization & Middleware Pipeline](#2-fastapi-initialization--middleware-pipeline)
3. [Backend Dependency Graph](#3-backend-dependency-graph)
4. [Authentication & Context Routing Mechanism](#4-authentication--context-routing-mechanism)
5. [Complete API Endpoint Catalogue](#5-complete-api-endpoint-catalogue)
   - [5.1 Authentication Router (`/api/auth`)](#51-authentication-router-apiauth)
   - [5.2 Emails & Analytics Router (`/api`)](#52-emails--analytics-router-api)
   - [5.3 Gmail Sync Router (`/api/gmail`)](#53-gmail-sync-router-apigmail)
   - [5.4 LLM Classification Router (`/api/llm`)](#54-llm-classification-router-apillm)
   - [5.5 RAG & AI Assistant Router (`/api/rag`)](#55-rag--ai-assistant-router-apirag)
   - [5.6 Export & Import Router (`/api/export`)](#56-export--import-router-apiexport)
   - [5.7 Developer Settings Router (`/api/developer`)](#57-developer-settings-router-apideveloper)
   - [5.8 System & SPA Fallback Routes](#58-system--spa-fallback-routes)
6. [Background Daemons & Workers](#6-background-daemons--workers)

---

## 1. Backend Overview & Entrypoint Architecture

The VC11 backend is a FastAPI application structured with a clear separation of concerns:
- **`app/main.py`**: Minimal script providing `import_emails()` for batch ingestion or CLI operations.
- **`app/api/app.py`**: The definitive web application factory and ASGI entrypoint (`uvicorn app.api.app:app`).
- **`app/core/config.py`**: Pydantic `Settings` class loading configuration from `.env`.
- **`app/core/security.py`**: HMAC-SHA256 session token verification and `get_current_user` FastAPI dependency.
- **`app/database/connection.py`**: Connection pool manager supporting dual-mode database routing (Central vs. Personal DB).
- **`app/database/crud.py`**: Low-level database persistence functions for users, accounts, and email data.
- **`app/services/`**: Specialized business logic services (Gmail API client, LLM categorization daemon, RAG embedding pipeline, export/import, scheduled delivery).
- **`app/rag/`**: LangGraph-based and LangChain-based mailbox RAG, 10 agent patterns, adaptive retrieval, and MCP tools.

---

## 2. FastAPI Initialization & Middleware Pipeline

```mermaid
flowchart TD
    REQ[Incoming HTTP Request] --> M1[route_developer_connections Middleware]
    M1 -->|Session authenticated?| CHECK_DEV{Has developer_database_url\nor developer_llm_api_key?}
    CHECK_DEV -->|Yes| CTX[Set database_url_context\nSet llm_key_context]
    CHECK_DEV -->|No| PASS1[Keep Central DB Default]
    CTX --> M2[CORSMiddleware]
    PASS1 --> M2
    M2 -->|Origins: frontend_url| M3[redirect_old_domain Middleware]
    M3 -->|Old Heroku Domain?| REDIR[301 Permanent Redirect]
    M3 -->|Normal Host| M4[protect_demo_mailbox Middleware]
    M4 -->|is_demo user & Mutating HTTP Method?| REJECT[403 Forbidden: Demo Read-Only]
    M4 -->|Safe / Authorized| ROUTE[FastAPI Route Dispatcher]
    ROUTE --> ENDPOINT[Route Handler Function]
```

### Lifespan Startup Daemons
When FastAPI starts up (`app/api/app.py:lifespan`):
1. **`init_db()` / `verify_schema()`**: Verifies that all 13 core tables exist in the target database (`users`, `gmail_accounts`, `emails`, `labels`, `email_labels`, `ai_categories`, `email_classifications`, `sync_history`, `llm_classification_jobs`, `chat_quotas`, `chat_concurrency_locks`, `chat_request_leases`, `email_send_approvals`).
2. **`start_global_ai_classifier_daemon()`**: Starts background worker polling for unclassified emails every 30 seconds.
3. **`start_global_rag_auto_indexer_daemon()`**: Starts background worker polling for un-embedded email chunks every 30 seconds.
4. **`start_global_email_delivery_daemon()`**: Starts APScheduler-style daemon checking every 15 seconds for approved emails where `scheduled_for <= now`.

---

## 3. Backend Dependency Graph

```mermaid
flowchart LR
    subgraph ROUTERS["API Routers (app/api/routes)"]
        AUTH_R[auth.py]
        EMAILS_R[emails.py]
        GMAIL_R[gmail.py]
        LLM_R[llm.py]
        RAG_R[rag.py]
        EXPORT_R[export.py]
        DEV_R[developer.py]
    end

    subgraph SERVICES["Business Logic (app/services)"]
        GMAIL_SVC[gmail_service.py]
        RAG_IDX[rag_indexer.py]
        RAG_AUTO[rag_auto_indexer.py]
        LLM_CLASS[llm_classification_service.py]
        SCHED[email_scheduler.py]
        EXP_SVC[export_service.py]
        DEV_SET[developer_settings.py]
        QUOTA[chat_quota.py]
        DEMO[demo_seed.py]
    end

    subgraph RAG_CORE["RAG & Agent System (app/rag)"]
        GRAPH[graph_agent.py]
        AGENT[agent.py]
        RETRIEVAL[retrieval.py]
        CHAINS[chains.py]
        TOOLS[tools.py]
        MCP[gmail_mcp.py]
        PGVEC[pgvector_store.py]
    end

    subgraph DB_MODELS["Database Layer (app/models)"]
        M_USER[User, GmailAccount, Email]
        M_RAG[EmailChunk, EmailEmbedding, ChatSession]
        M_LLM[LlmClassificationJob]
        M_QUOTA[ChatQuota, ChatConcurrencyLock]
    end

    AUTH_R --> GMAIL_SVC & DEV_SET & DEMO & M_USER
    EMAILS_R --> GMAIL_SVC & LLM_CLASS & M_USER
    GMAIL_R --> GMAIL_SVC & LLM_CLASS & RAG_AUTO & M_USER
    LLM_R --> LLM_CLASS & M_LLM & M_USER
    RAG_R --> GRAPH & AGENT & TOOLS & QUOTA & GMAIL_SVC & M_RAG
    EXPORT_R --> EXP_SVC & M_USER
    DEV_R --> DEV_SET & M_USER

    GRAPH --> AGENT & TOOLS & RETRIEVAL & CHAINS & MCP
    RETRIEVAL --> PGVEC & M_RAG
    TOOLS --> GMAIL_SVC & M_RAG & M_USER
    RAG_AUTO --> RAG_IDX --> PGVEC
```

---

## 4. Authentication & Context Routing Mechanism

Authentication is completely cookie-based:
- **Cookie Name**: `gmail_ai_session`
- **Format**: `<base64url_json_payload>.<hmac_sha256_signature>`
- **Payload**: `{"user_id": "<uuid>", "exp": <unix_timestamp>}`
- **Signature**: `HMAC-SHA256(payload, settings.session_secret)`
- **Duration**: 7 days for normal users; 1 day for demo users.
- **Security**: `HttpOnly=True`, `SameSite=Lax`, `Secure` in production.

Whenever an API request executes:
1. `get_current_user` extracts and verifies the HMAC signature using `settings.session_secret`.
2. Queries the user from the database.
3. If `user.developer_database_url` is present, `route_developer_connections` decrypts it using Fernet and binds it to `database_url_context`.
4. `get_db()` yields `SessionLocal()`, which targets the user's personal database if set, or the central deployment database otherwise.

---

## 5. Complete API Endpoint Catalogue

### 5.1 Authentication Router (`/api/auth`)
*Source file:* `backend/app/api/routes/auth.py`

#### `GET /api/auth/google`
- **Description:** Initiates Google OAuth2 login flow.
- **Auth Required:** No.
- **Query Params:** None.
- **Returns:** 302 Redirect to `https://accounts.google.com/o/oauth2/v2/auth` with Google OAuth parameters (`access_type=offline`, `prompt=consent`, `response_type=code`).
- **Cookies Set:** `gmail_ai_oauth_state` (CSRF token, 10 min), `gmail_ai_oauth_verifier` (PKCE code verifier, 10 min).
- **Frontend Caller:** `window.location.assign(getGoogleAuthUrl())` in `GmailConsent.jsx`.

#### `GET /api/auth/callback` (alias: `GET /auth/google/callback`)
- **Description:** Handles Google OAuth2 callback, verifies code, exchanges for tokens, and creates/updates user session.
- **Auth Required:** No (validates state & PKCE cookies).
- **Query Params:** `code: str`, `state: str`.
- **Services Called:** Google OAuth token endpoint, `google.oauth2.id_token.verify_oauth2_token()`, `upsert_google_user()`, `ensure_personal_identity()`, `ensure_user_emails_embedded()`.
- **Tables Accessed:** `users`, `gmail_accounts`.
- **Returns:** 302 Redirect to `{frontend_url}/sync` with `gmail_ai_session` cookie (7 days).
- **Errors:** 400 (Invalid callback/state mismatch/expired code), 409 (Attempted login with reserved demo email).

#### `GET /api/auth/me`
- **Description:** Returns the profile and developer settings status for the authenticated user.
- **Auth Required:** Yes (`gmail_ai_session`).
- **Handler:** `current_user(user: User = Depends(get_current_user))`
- **Returns:**
  ```json
  {
    "id": "uuid",
    "email": "user@example.com",
    "name": "User Name",
    "picture": "https://...",
    "is_demo": false,
    "has_personal_database": false,
    "has_personal_llm_key": false,
    "ai_classification_enabled": true
  }
  ```
- **Frontend Caller:** `getCurrentUser()` in `frontend/src/services/api.js` (called on mount by `AuthContext.jsx`).

#### `POST /api/auth/demo`
- **Description:** Logs the user into the seeded, read-only demo mailbox without Google OAuth.
- **Auth Required:** No.
- **Tables Accessed:** `users` (queries `google_id = "demo:mailbox"`).
- **Returns:** Serialized demo user profile JSON and sets `gmail_ai_session` (1 day) & `gmail_ai_demo_visitor` cookies.
- **Frontend Caller:** `enterDemo()` in `frontend/src/services/api.js` (used in `PublicPages.jsx` and `App.jsx`).

#### `POST /api/auth/logout`
- **Description:** Terminates the session.
- **Status:** 204 No Content.
- **Returns:** Clears `gmail_ai_session` cookie.
- **Frontend Caller:** `logout()` in `frontend/src/services/api.js`.

#### `DELETE /api/auth/erase`
- **Description:** Erases all synchronized emails, labels, chunks, embeddings, and chat history for the user while preserving the account row.
- **Auth Required:** Yes.
- **Status:** 204 No Content.
- **Services Called:** `delete_user_gmail_data(db, user)`.
- **Tables Accessed:** Deletes from `emails`, `labels`, `email_labels`, `email_classifications`, `email_chunks`, `email_embeddings`, `chat_sessions`, `sync_history`, `email_send_approvals`.
- **Frontend Caller:** `eraseAllData()` in `frontend/src/services/api.js` (Settings Page Danger Zone).

---

### 5.2 Emails & Analytics Router (`/api`)
*Source file:* `backend/app/api/routes/emails.py`

#### `GET /api/analytics/charts`
- **Description:** Computes aggregated email distribution by AI category and time-series activity for dashboard visualizations.
- **Auth Required:** Yes.
- **Query Params:** `range: str` (e.g. `today`, `this_week`, `this_month`, `six_months`, `one_year`, `all_time`), `tz: str` (timezone).
- **Headers:** `X-Timezone: str`.
- **Tables Accessed:** `emails`, `email_labels`, `labels`, `email_classifications`, `ai_categories`.
- **Returns:**
  ```json
  {
    "range": "this_month",
    "label": "This month",
    "total_emails": 1248,
    "categories": [
      { "name": "Work", "value": 42, "count": 524 },
      { "name": "Personal", "value": 28, "count": 350 }
    ],
    "activity": [
      { "label": "Oct 1", "incoming": 15, "outgoing": 7 }
    ]
  }
  ```
- **Frontend Caller:** `fetchAnalyticsCharts(range, tz)` in `App.jsx`.

#### `GET /api/analytics/summary`
- **Description:** Calculates comparative KPI summary metrics (total, auto-labeled, important, unread) against previous period.
- **Auth Required:** Yes.
- **Query Params:** `range: str`, `tz: str`.
- **Returns:**
  ```json
  {
    "range": "this_week",
    "label": "This week",
    "current": { "total": 120, "auto_labeled": 95, "important": 18, "unread": 12 },
    "comparison": { "total": 105, "auto_labeled": 80, "important": 14, "unread": 20 }
  }
  ```
- **Frontend Caller:** `fetchAnalyticsSummary(range, tz)` in `App.jsx`.

#### `GET /api/emails`
- **Description:** Paginated email listing with multi-attribute filtering.
- **Auth Required:** Yes.
- **Query Params:**
  - `page: int = 1`
  - `per_page: int = 50`
  - `search: Optional[str]`
  - `label: Optional[str]`
  - `inbox_only: bool = False`
  - `sent_only: bool = False`
  - `unread_only: bool = False`
  - `range: Optional[str]`
  - `tz: Optional[str]`
- **Tables Accessed:** `emails`, `email_labels`, `labels`, `email_classifications`.
- **Returns:**
  ```json
  {
    "total": 524,
    "page": 1,
    "per_page": 50,
    "emails": [
      {
        "id": "uuid",
        "gmail_message_id": "189...",
        "thread_id": "189...",
        "from": "alice@example.com",
        "from_email": "alice@example.com",
        "sender_name": "Alice Smith",
        "to": "user@example.com",
        "subject": "Project Status",
        "snippet": "Here is the weekly update...",
        "received_at": "2026-10-30T10:00:00Z",
        "is_read": true,
        "is_starred": false,
        "has_attachment": false,
        "labels": ["INBOX", "WORK"],
        "category": "WORK"
      }
    ]
  }
  ```
- **Frontend Caller:** `fetchEmails(params)` and `fetchSentEmails(params)` in `api.js`.

#### `GET /api/emails/{email_id}`
- **Description:** Retrieves full message details, body text, sanitized HTML, and attachment metadata. If HTML is missing locally, hydrates live from Gmail API.
- **Auth Required:** Yes.
- **External Calls:** Gmail API `users.messages.get` (if HTML not cached locally).
- **Frontend Caller:** `fetchEmail(id)` in `App.jsx`.

#### `GET /api/labels`
- **Description:** Lists all Gmail labels associated with the user's account.
- **Auth Required:** Yes.
- **Tables Accessed:** `labels`.
- **Frontend Caller:** `fetchLabels()` in `api.js`.

#### `POST /api/emails/send` (alias: `POST /api/gmail/send`)
- **Description:** Directly sends an outbound email via Gmail API or simulates delivery in demo mode.
- **Auth Required:** Yes.
- **Request Body:**
  ```json
  {
    "to": "recipient@example.com",
    "cc": "optional@example.com",
    "bcc": "hidden@example.com",
    "subject": "Subject Line",
    "body": "Message plain text or draft content"
  }
  ```
- **External Calls:** Gmail API `users.messages.send`.
- **Tables Accessed:** `emails` (inserts sent email cache), `email_labels` (associates `SENT` label).
- **Frontend Caller:** `sendEmail(data)` in `ComposeModal.jsx`.

---

### 5.3 Gmail Sync Router (`/api/gmail`)
*Source file:* `backend/app/api/routes/gmail.py`

#### `POST /api/gmail/sync`
- **Description:** Triggers background synchronization of Gmail mailbox.
- **Auth Required:** Yes (non-demo users only).
- **Concurrency Control:** Checks 15-minute sync heartbeat lease on `gmail_accounts.inbox_sync_heartbeat_at`.
- **Worker:** Spawns a background daemon thread running `_run_sync_in_thread`.
- **Returns:** `{"message": "Sync started", "status": "syncing", "stage": "starting", ...}`
- **Frontend Caller:** `startSync()` in `SyncPage.jsx` and `App.jsx`.

#### `GET /api/gmail/sync/status`
- **Description:** Polls current state of Gmail synchronization.
- **Auth Required:** Yes.
- **Returns:**
  ```json
  {
    "status": "syncing",
    "stage": "fetching",
    "emails_found": 500,
    "emails_synced": 320,
    "labels_synced": 15,
    "background_status": "idle",
    "error": null
  }
  ```
- **Frontend Caller:** `getSyncStatus()` in `SyncPage.jsx` and `App.jsx`.

#### `GET /api/gmail/sync/stream`
- **Description:** Server-Sent Events (SSE) streaming real-time sync progress.
- **Auth Required:** Yes.
- **Media Type:** `text/event-stream`.
- **Event Payload:** `data: {"status": "syncing", "stage": "storing", ...}\n\n`
- **Frontend Caller:** `new EventSource(getSyncStreamUrl())` in `SyncPage.jsx`.

---

### 5.4 LLM Classification Router (`/api/llm`)
*Source file:* `backend/app/api/routes/llm.py`

#### `GET /api/llm/preference`
- **Description:** Reads user's AI categorization toggle.
- **Returns:** `{"ai_classification_enabled": true, "gmail_account_email": "user@gmail.com"}`

#### `PUT /api/llm/preference`
- **Description:** Updates AI categorization preference.
- **Query / Body:** `enabled: bool`.

#### `POST /api/llm/classify/backfill` (aliases: `/classify/start`, `/classify/all`, `/train`)
- **Description:** Enqueues a background LLM classification job for all unclassified emails.
- **Status:** 202 Accepted.
- **Services Called:** `llm_classification_service.enqueue_classification_job(account.id, "MANUAL_TRAIN")`.
- **Returns:** `{"message": "AI classification started...", "job_id": "uuid", "emails_discovered": 120}`
- **Frontend Caller:** `trainModel()` in `api.js` (Quick Actions in `App.jsx`).

#### `POST /api/llm/classify/retry`
- **Description:** Retries the latest failed or partially completed classification job.
- **Status:** 202 Accepted.

#### `GET /api/llm/classify/status`
- **Description:** Returns live in-memory worker state and latest database job record.

#### `GET /api/llm/classify/jobs`
- **Description:** Returns the last 20 classification jobs for the user.
- **Tables Accessed:** `llm_classification_jobs`.

---

### 5.5 RAG & AI Assistant Router (`/api/rag`)
*Source file:* `backend/app/api/routes/rag.py`

#### `POST /api/rag/chat`
- **Description:** Non-streaming AI Inbox Assistant endpoint executing the LangGraph multi-pattern workflow.
- **Auth Required:** Yes.
- **Rate Limit & Concurrency:** Protected by `chat_quota.py` (daily limit 50 for regular users, 500 for developers; concurrency limit 5).
- **Request Body (`ChatRequest`):**
  ```json
  {
    "query": "Find all interview invitations received in the last 2 weeks",
    "session_id": "optional-uuid",
    "top_k": 5,
    "filter_dict": { "label": "INBOX" },
    "sender": "recruiter@company.com",
    "use_multi_query": true,
    "use_gmail_mcp": true
  }
  ```
- **Returns (`ChatResponse`):**
  ```json
  {
    "answer": "You received 2 interview invitations...",
    "intent": "RAG_QUERY",
    "citations": [
      {
        "email_id": "uuid",
        "subject": "Interview with Anthropic",
        "sender": "recruiter@anthropic.com",
        "date": "2026-10-28T14:30:00Z",
        "snippet": "We would like to invite you...",
        "score": 0.89,
        "key_points": ["Interview scheduled for Nov 2 at 10 AM PST."]
      }
    ],
    "steps": [...],
    "plan": ["Search for interview emails", "Extract dates", "Synthesize summary"],
    "approval_required": false,
    "approval_id": null,
    "session_id": "uuid",
    "message_id": "uuid"
  }
  ```

#### `POST /api/rag/chat/stream`
- **Description:** High-performance Server-Sent Events (SSE) streaming endpoint for the AI Assistant. Provides real-time execution steps, tool activities, and grounded answers.
- **Auth Required:** Yes.
- **Media Type:** `text/event-stream`.
- **Stream Events:**
  - `event: activity` → Emitted at each pipeline step (e.g., `route_request`, `gmail_search`, `validate_evidence`, `compose_answer`). Rendered live in frontend `SearchActivity.jsx`.
  - `event: result` → Final `ChatResponse` JSON payload.
  - `event: error` → Error details if generation fails.
- **Frontend Caller:** `sendRagChat(data, onProgress)` in `frontend/src/services/api.js`.

#### `POST /api/rag/index`
- **Description:** Triggers background incremental or full PostgreSQL pgvector re-indexing for the user's emails.
- **Auth Required:** Yes.
- **Request Body:** `{"batch_size": 100, "full_reindex": false}`.
- **Services Called:** `app.rag.indexing.default_pipeline.index_user_emails()`.
- **Frontend Caller:** `triggerRagIndex()` in `SettingsPage` (`App.jsx`).

#### `GET /api/rag/status`
- **Description:** Returns diagnostic statistics on chunking and vector indexing.
- **Returns:**
  ```json
  {
    "total_emails": 1248,
    "total_chunks": 3120,
    "total_embeddings": 3120,
    "indexed_embeddings": 3120,
    "pending_embeddings": 0,
    "pending_approvals": 1,
    "vector_backend": "postgres",
    "dimensions": 3072
  }
  ```
- **Frontend Caller:** `getRagStatus()` in `App.jsx` and `AiAssistant.jsx`.

#### `GET /api/rag/sessions`
- **Description:** Lists user's previous AI conversation sessions (up to limit).
- **Tables Accessed:** `chat_sessions`.
- **Frontend Caller:** `getRagSessions(limit)` in `AiAssistant.jsx`.

#### `GET /api/rag/sessions/{session_id}`
- **Description:** Retrieves message history and citations for a specific conversation session.
- **Tables Accessed:** `chat_sessions`, `chat_messages`, `chat_retrievals`, `email_chunks`.
- **Frontend Caller:** `getRagSession(sessionId)` in `AiAssistant.jsx`.

#### `DELETE /api/rag/sessions/{session_id}`
- **Description:** Deletes a chat session and all associated message/retrieval records.
- **Frontend Caller:** `deleteRagSession(sessionId)` in `AiAssistant.jsx`.

#### `GET /api/rag/approvals`
- **Description:** Lists outbound emails pending human approval.
- **Tables Accessed:** `email_send_approvals`.
- **Frontend Caller:** `getRagApprovals(statusFilter)` in `api.js`.

#### `PATCH /api/rag/approvals/{approval_id}`
- **Description:** Updates the draft fields (recipient, CC, BCC, subject, body) of a pending email draft without sending it.
- **Tables Accessed:** `email_send_approvals`.
- **Frontend Caller:** `updateRagEmailDraft(approvalId, draft)` in `AiAssistant.jsx`.

#### `POST /api/rag/approvals/{approval_id}/approve`
- **Description:** Human-in-the-Loop approval gate. Sends the approved email via Gmail API or schedules it.
- **Status Change:** `PENDING` → `APPROVED` → `SENT` (or `SCHEDULED`).
- **External Calls:** `send_gmail_message()`.
- **Frontend Caller:** `approveRagEmail(approvalId, draft)` in `AiAssistant.jsx`.

#### `POST /api/rag/approvals/{approval_id}/reject`
- **Description:** Discards a drafted email.
- **Status Change:** `PENDING` → `REJECTED`.
- **Frontend Caller:** `rejectRagEmail(approvalId)` in `AiAssistant.jsx`.

#### `GET /api/rag/sources/{email_id}`
- **Description:** Citation inspector endpoint returning cleaned email text, full HTML, and attachment references for any email cited in AI responses.
- **Frontend Caller:** `getRagSourceEmail(emailId)` in `EmailDetailsModal.jsx`.

---

### 5.6 Export & Import Router (`/api/export`)
*Source file:* `backend/app/api/routes/export.py`

#### `GET /api/export/stats`
- **Description:** Returns counts and dates for downloadable emails.
- **Returns:** `{"total_emails": 1248, "emails_with_body": 1248, "labels_count": 18, "oldest_email": "...", "newest_email": "..."}`
- **Frontend Caller:** `getExportStats()` in `SettingsPage` (`App.jsx`).

#### `GET /api/export/csv`
- **Description:** Downloads synchronized emails as a CSV file.
- **Query Params:** `include_body: bool = True`, `limit: Optional[int]`.
- **Frontend Caller:** `getExportCSVUrl(params)` in `App.jsx`.

#### `GET /api/export/json`
- **Description:** Downloads synchronized emails as a JSON file.
- **Query Params:** `include_body: bool = True`, `include_html: bool = False`, `limit: Optional[int]`.
- **Frontend Caller:** `getExportJSONUrl(params)` in `App.jsx`.

#### `POST /api/export/import`
- **Description:** Uploads a JSON or CSV file to import emails into the database.
- **Request Format:** `multipart/form-data` with `file` upload.
- **Frontend Caller:** `importEmailsFile(formData)` in `SettingsPage` (`App.jsx`).

---

### 5.7 Developer Settings Router (`/api/developer`)
*Source file:* `backend/app/api/routes/developer.py`

#### `GET /api/developer`
- **Description:** Reads developer settings (personal database configuration, custom LLM provider, quota limits). Note: Passwords and API keys are never returned in plaintext.
- **Auth Required:** Yes (non-demo users only).
- **Returns:**
  ```json
  {
    "database_configured": true,
    "database_url_display": "postgresql://user:***@db.host.com:5432/mydb",
    "llm_provider": "google",
    "llm_key_configured": true,
    "daily_chat_limit": 500
  }
  ```
- **Frontend Caller:** `getDeveloperSettings()` in `SettingsPage` (`App.jsx`).

#### `PUT /api/developer`
- **Description:** Configures or disconnects a personal PostgreSQL database or custom LLM API key.
- **Request Body (`DeveloperSettingsUpdate`):**
  ```json
  {
    "database_url": "postgresql://...",
    "llm_provider": "google",
    "llm_api_key": "AIza...",
    "clear_database": false,
    "clear_llm_key": false
  }
  ```
- **Logic:**
  - If `database_url` is provided: normalizes to `postgresql+psycopg://`, tests connection and schema completeness via `verify_personal_database()`, encrypts with Fernet, and saves.
  - If `clear_database: true`: removes personal database and reverts to central application database.
- **Frontend Caller:** `updateDeveloperSettings(changes)` in `SettingsPage` (`App.jsx`).

#### `GET /api/developer/schema.sql`
- **Description:** Generates on-the-fly DDL SQL script reflecting all SQLAlchemy models for initializing personal databases.
- **Returns:** File attachment `gmail-ai-dashboard-schema.sql`.
- **Frontend Caller:** `getDeveloperSchemaUrl()` in `SettingsPage` (`App.jsx`).

---

### 5.8 System & SPA Fallback Routes
*Source file:* `backend/app/api/app.py`

#### `GET /health`
- **Description:** Health check endpoint for container orchestrators (Heroku dynos, Docker).
- **Auth Required:** No.
- **Returns:** `{"status": "ok"}`.

#### `GET /{path:path}`
- **Description:** Serves static assets from `frontend/dist/` or falls back to `index.html` for client-side React Router navigation.

---

## 6. Background Daemons & Workers

| Daemon | File | Interval | Trigger | Responsibility |
|---|---|---|---|---|
| **AI Classifier** | `services/llm_classification_service.py` | 30s | Startup + on sync/import | Discovers unclassified emails, calls LLM, saves `email_classifications`. |
| **RAG Auto-Indexer** | `services/rag_auto_indexer.py` | 30s | Startup + on sync/login | Finds emails lacking embeddings, runs `chunk_email()`, generates vectors, saves `email_embeddings`. |
| **Email Delivery Scheduler** | `services/email_scheduler.py` | 15s | Startup | Queries `email_send_approvals` where `status='SCHEDULED'` and `scheduled_for <= now`, sends via Gmail API. |
| **Inbox Synchronizer** | `api/routes/gmail.py` | On-demand thread | `POST /api/gmail/sync` | Paginates Gmail API, fetches full messages, upserts into `emails` and `labels`. |

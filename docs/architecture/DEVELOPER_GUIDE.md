# VC11 — Developer Onboarding, Code Navigation & Implementation Guide

> **Audience:** Software Engineers, Full-Stack Developers, AI/ML Engineers, and System Architects contributing to or onboarding onto the VC11 (Gmail AI Dashboard) codebase.

---

## Table of Contents

1. [Developer Mental Model](#1-developer-mental-model)
2. [Repository Code Map & Responsibility Matrix](#2-repository-code-map--responsibility-matrix)
3. [End-to-End Code Walkthroughs](#3-end-to-end-code-walkthroughs)
   - [3.1 Walkthrough A: What happens when a user clicks "Sync Gmail"](#31-walkthrough-a-what-happens-when-a-user-clicks-sync-gmail)
   - [3.2 Walkthrough B: What happens when a user sends a prompt to the AI Assistant](#32-walkthrough-b-what-happens-when-a-user-sends-a-prompt-to-the-ai-assistant)
   - [3.3 Walkthrough C: What happens when the AI drafts an email and the user approves it](#33-walkthrough-c-what-happens-when-the-ai-drafts-an-email-and-the-user-approves-it)
   - [3.4 Walkthrough D: What happens when a user connects a Personal Database](#34-walkthrough-d-what-happens-when-a-user-connects-a-personal-database)
4. [Local Development & Debugging Recipes](#4-local-development--debugging-recipes)
   - [4.1 Running the Full Stack Locally](#41-running-the-full-stack-locally)
   - [4.2 Debugging SSE Telemetry Streams](#42-debugging-sse-telemetry-streams)
   - [4.3 Inspecting Background Daemons](#43-inspecting-background-daemons)
   - [4.4 LangSmith Tracing Setup](#44-langsmith-tracing-setup)
5. [Developer How-To Guides](#5-developer-how-to-guides)
   - [5.1 How to Add a New Agent Tool](#51-how-to-add-a-new-agent-tool)
   - [5.2 How to Create a New Database Migration](#52-how-to-create-a-new-database-migration)
   - [5.3 How to Add a New API Endpoint](#53-how-to-add-a-new-api-endpoint)
6. [Testing & Verification](#6-testing--verification)

---

## 1. Developer Mental Model

To understand VC11 quickly, think of it in three primary loops:

```mermaid
flowchart LR
    subgraph INGESTION_LOOP["1. Ingestion Loop (Background)"]
        GMAIL_API[Gmail API] --> SYNC_SVC[gmail_service.py]
        SYNC_SVC --> DB_EMAILS[(emails Table)]
        DB_EMAILS --> RAG_AUTO[rag_auto_indexer.py]
        RAG_AUTO --> CHUNKS[(email_chunks)]
        CHUNKS --> EMBEDDINGS[(email_embeddings pgvector)]
    end

    subgraph ASSISTANT_LOOP["2. AI Assistant Loop (Interactive)"]
        PROMPT[User Prompt] --> STREAM_API[/api/rag/chat/stream]
        STREAM_API --> GRAPH[LangGraph graph_agent.py]
        GRAPH --> RETRIEVE[HybridRetriever: pgvector + BM25]
        RETRIEVE --> CITATIONS[Grounded Citations]
        CITATIONS --> ANSWER[Answer Generation]
        ANSWER --> SSE_OUT[SSE Activity & Result Stream]
    end

    subgraph ACTION_LOOP["3. Action & Safety Loop (HITL)"]
        DRAFT[Agent Drafts Email] --> APPROVAL[(email_send_approvals Table)]
        APPROVAL --> UI_CARD[Approval Card in UI]
        UI_CARD -->|User Clicks Approve| GMAIL_SEND[send_gmail_message]
        GMAIL_SEND --> GMAIL_DELIVER[Gmail API Delivery]
    end
```

1. **The Ingestion Loop (Asynchronous):** Syncs messages from the Gmail API into PostgreSQL, strips HTML, breaks messages into overlapping 1000-character chunks, generates 3072-dimensional vector embeddings via Azure OpenAI, and indexes them using pgvector (`HALFVEC(3072)`).
2. **The Assistant Loop (Real-Time):** User prompts trigger a stateful LangGraph agent that classifies intent, rewrites queries, plans search strategies, executes hybrid retrieval (dense pgvector cosine similarity + sparse BM25 with $3\times$ subject boosting fused via RRF), validates evidence, and streams real-time execution steps and grounded answers via Server-Sent Events (SSE).
3. **The Action Loop (Human-in-the-Loop):** Autonomous actions (drafting emails) create `PENDING` records in `email_send_approvals`. Outbound emails are **never transmitted** until the user reviews and confirms them through the UI.

---

## 2. Repository Code Map & Responsibility Matrix

| Path | Primary Responsibility | Key Files |
|---|---|---|
| `backend/app/api/` | Application entrypoint, ASGI initialization, middleware, routes | `app.py`, `dependencies.py` |
| `backend/app/api/routes/`| FastAPI route controllers | `auth.py`, `emails.py`, `gmail.py`, `llm.py`, `rag.py`, `export.py`, `developer.py` |
| `backend/app/core/` | Configuration loading and HMAC authentication security | `config.py`, `security.py` |
| `backend/app/database/` | Database engine, connection pooling, and CRUD | `connection.py`, `crud.py` |
| `backend/app/models/` | SQLAlchemy declarative database models | `email.py`, `rag.py`, `llm.py`, `chat_quota.py` |
| `backend/app/rag/` | LangGraph agent graph, 18 tools, hybrid retrieval, BM25, MCP bridge | `graph_agent.py`, `agent.py`, `tools.py`, `retrieval.py`, `adaptive.py`, `chains.py`, `gmail_mcp.py` |
| `backend/app/services/` | Business logic daemons and external integrations | `gmail_service.py`, `rag_auto_indexer.py`, `llm_classification_service.py`, `email_scheduler.py`, `developer_settings.py`, `gmail_mcp_server.py` |
| `backend/alembic/` | Database schema migrations (16 versioned revisions) | `env.py`, `versions/*.py` |
| `frontend/src/` | React single page application | `App.jsx`, `main.jsx` |
| `frontend/src/pages/` | Primary views and workflows | `AiAssistant.jsx`, `SyncPage.jsx`, `GmailConsent.jsx`, `PublicPages.jsx` |
| `frontend/src/components/`| UI components and modals | `ComposeModal.jsx`, `EmailDetailsModal.jsx`, `SearchActivity.jsx`, `SentMessagesView.jsx` |
| `frontend/src/services/` | Centralized Axios and SSE API client | `api.js` |
| `frontend/src/context/` | Global session and authentication state | `AuthContext.jsx` |
| `docs/architecture/` | Complete architectural documentation suite | `VC11_MASTER_ARCHITECTURE.md`, `VC11_LOW_LEVEL_ARCHITECTURE.md`, etc. |

---

## 3. End-to-End Code Walkthroughs

### 3.1 Walkthrough A: What happens when a user clicks "Sync Gmail"
1. **User Action:** User clicks "Sync Gmail" in `QuickActions` (`frontend/src/App.jsx`).
2. **Frontend Call:** `handleSync()` invokes `startSync()` (`POST /api/gmail/sync`) from `frontend/src/services/api.js`.
3. **Route Handling:** `backend/app/api/routes/gmail.py:start_sync` receives the request.
   - Evaluates `_sync_lease_fresh()`.
   - Executes optimistic database lock: updates `gmail_accounts.sync_status = 'INITIAL'` and `inbox_sync_heartbeat_at = now()`.
   - Copies request context via `contextvars.copy_context()`.
   - Spawns background thread executing `_run_sync_in_thread(user_id_str, user_id)`.
4. **Synchronization Logic (`backend/app/services/gmail_service.py`):**
   - Refreshes expired OAuth tokens via `build_gmail_service_for_user()`.
   - Calls `users.messages.list(maxResults=500)` to enumerate messages.
   - Batches full message retrieval via `users.messages.get(id, format='full')`.
   - Parses MIME multipart headers and bodies via `_parse_message()`.
   - Upserts records into `emails` and `email_labels` tables.
   - Updates `gmail_accounts.sync_status = 'BACKFILL'` and records history in `sync_history`.
5. **Background Daemons Triggered:**
   - Enqueues initial LLM classification job (`llm_classification_service.enqueue_classification_job`).
   - Triggers RAG auto-indexing (`rag_auto_indexer.ensure_user_emails_embedded`).
6. **Frontend Update:** Frontend opens SSE stream `GET /api/gmail/sync/stream` to receive live progress counters (`emails_found`, `emails_synced`), updating the UI until `status == 'completed'`.

---

### 3.2 Walkthrough B: What happens when a user sends a prompt to the AI Assistant
1. **User Action:** User types *"When is my flight to Chicago?"* and presses Enter in `AiAssistant.jsx`.
2. **Frontend Call:** `handleSend()` invokes `sendRagChat({ query, session_id, top_k: 5 }, onProgress)` in `services/api.js`.
   - Connects to `POST /api/rag/chat/stream` using native `fetch` and initializes a `TextDecoder` reader.
3. **Route & Concurrency Reservation (`backend/app/api/routes/rag.py`):**
   - Calls `reserve_chat_request(db, request, user)` in `chat_quota.py`.
   - Acquires row lock `SELECT ... FOR UPDATE` on `chat_concurrency_locks`.
   - Checks daily quota (50 standard / 500 developer) and concurrency ceiling (max 5).
   - Issues a `ChatConcurrencyLease` with a background 30-second heartbeat thread.
4. **LangGraph Graph Execution (`backend/app/rag/graph_agent.py`):**
   - **`route_request` node:** Uses `ROUTER_PROMPT` to classify query as `RAG_QUERY`.
   - **`plan_search` node:** Rewrites query into standalone search string and chooses retrieval strategy (`HybridRetriever`).
   - **`retrieve_candidates` node:**
     - Query embedding generated via Azure OpenAI `text-embedding-3-large`.
     - Dense similarity search executed via pgvector: `ORDER BY embedding <=> query_embedding LIMIT 20`.
     - Lexical candidate search executed via BM25 with $3\times$ subject boost over cached mailbox text.
     - Merges candidates using Reciprocal Rank Fusion (RRF with $k=60$).
   - **`validate_evidence` node:** Checks semantic relevance against retrieved chunks.
   - **`compose_answer` node:** Prompts LLM with grounded email evidence, instructing it to cite sources (`[1]`, `[2]`).
   - **`persist_turn` node:** Saves assistant response to `chat_messages` and logs cited chunk IDs in `chat_retrievals`.
5. **SSE Telemetry:** Emits `event: activity` frames for each node turn, updating the live `SearchActivity` timeline in the frontend, followed by `event: result` containing the final answer and citation pills.

---

### 3.3 Walkthrough C: What happens when the AI drafts an email and the user approves it
1. **User Prompt:** *"Draft an email to john@example.com confirming our Tuesday 10am meeting."*
2. **Agent Detection:**
   - Router detects intent `ACTION_DRAFT`.
   - LLM extracts recipient, subject, and body via `parse_draft_request()`.
   - Inserts row in `email_send_approvals` with `status = "PENDING"`.
   - Returns `AgentResponse` with `approval_required = True`, `approval_id`, and `approval_preview`.
3. **UI Rendering:** `AiAssistant.jsx` detects `approval_required: true` and renders the **Approval Card** with "Review subject & body", "Approve & send", and "Discard draft" buttons.
4. **User Approval:** User clicks "Approve & send" $\rightarrow$ triggers `POST /api/rag/approvals/{approval_id}/approve`.
5. **Execution (`backend/app/api/routes/rag.py:approve_email_send`):**
   - Validates that approval record exists, belongs to the authenticated user, and has `status == 'PENDING'`.
   - Calls `send_gmail_message()` in `gmail_service.py`.
   - Encodes MIME email and calls Gmail API `service.users().messages().send()`.
   - Updates `email_send_approvals` to `status = 'SENT'`, records `delivery_message_id` and `sent_at`.
   - Inserts sent email record into `emails` with `SENT` label.
   - Frontend removes approval card and confirms delivery.

---

### 3.4 Walkthrough D: What happens when a user connects a Personal Database
1. **User Action:** In `/settings`, user enters a PostgreSQL URL in Developer Settings and clicks "Connect Database".
2. **API Call:** `PUT /api/developer` with `{ database_url: "postgresql://..." }`.
3. **Route Handling (`backend/app/api/routes/developer.py`):**
   - Normalizes driver string: `postgresql+psycopg://`.
   - Calls `verify_personal_database(url)`:
     - Resolves hostname via DNS, blocks private/local IP ranges (SSRF mitigation), and pins IP.
     - Connects with a strict 5-second timeout and `NullPool`.
     - Inspects tables and verifies all required Alembic tables exist.
   - Encrypts database URL at rest using Fernet with `DEVELOPER_SETTINGS_ENCRYPTION_KEY`.
   - Updates `users.developer_database_url` in the central database.
4. **Subsequent API Requests:**
   - Middleware `route_developer_connections` decrypts `developer_database_url`.
   - Binds it to `database_url_context` (`ContextVar`).
   - Every invocation of `SessionLocal()` automatically targets the personal database.
   - **Failure Behavior:** If the personal database connection drops, the request fails immediately (`503 Service Unavailable`). **It never silently falls back to the central database.**

---

## 4. Local Development & Debugging Recipes

### 4.1 Running the Full Stack Locally

#### Terminal 1 — Backend:
```bash
cd gmail-ai-dashboard/backend
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Verify database schema is up-to-date
alembic upgrade head

# Run Uvicorn in auto-reload mode
uvicorn app.api.app:app --reload --port 8000
```

#### Terminal 2 — Frontend:
```bash
cd gmail-ai-dashboard/frontend
npm install
npm run dev
```

---

### 4.2 Debugging SSE Telemetry Streams

You can monitor Server-Sent Events directly from your terminal using `curl`:

```bash
# Stream AI assistant turn telemetry:
curl -N -X POST http://localhost:8000/api/rag/chat/stream \
  -H "Content-Type: application/json" \
  -H "Cookie: gmail_ai_session=<your_cookie>" \
  -d '{"query": "Summarize unread emails from today", "top_k": 5}'
```

You will see the live event stream:
```text
event: activity
data: {"action": "route_request", "label": "Routing request intent", "status": "running"}

event: activity
data: {"action": "retrieve_candidates", "label": "Searching indexed vector embeddings", "status": "running"}

event: result
data: {"answer": "...", "citations": [...], "session_id": "..."}
```

---

### 4.3 Inspecting Background Daemons

The application runs 3 background daemons started during FastAPI lifespan:
- **Classifier Daemon:** Check `backend/app/services/llm_classification_service.py`. It polls for unclassified emails every 30 seconds.
- **RAG Auto-Indexer:** Check `backend/app/services/rag_auto_indexer.py`. It polls for un-embedded chunks every 30 seconds.
- **Email Delivery Scheduler:** Check `backend/app/services/email_scheduler.py`. It polls for approved emails where `scheduled_for <= now` every 15 seconds.

Logs from these daemons appear with `[app.services.*]` prefixes in your Uvicorn console.

---

### 4.4 LangSmith Tracing Setup

To visualize LangGraph node executions, tool arguments, and LLM latencies:
1. Obtain an API key from [smith.langchain.com](https://smith.langchain.com).
2. Add the following to `backend/.env`:
   ```env
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=lsv2_pt_...
   LANGSMITH_PROJECT=mailbox-devLG
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   ```
3. Restart the backend. Every chat query will log a root trace in your LangSmith project dashboard.

---

## 5. Developer How-To Guides

### 5.1 How to Add a New Agent Tool
1. Open `backend/app/rag/tools.py`.
2. Define a function decorated with `@tool`:
   ```python
   @tool
   def search_unread_newsletters(category: str = "NEWSLETTERS") -> str:
       """Search for unread newsletters received this week."""
       # Always enforce user scoping:
       # user_uuid is available in create_gmail_tools(db, user_id)
       ...
       return json.dumps(results)
   ```
3. Add the function to the `tools` list in `create_gmail_tools(db, user_id)`:
   ```python
   tools = [
       ...
       search_unread_newsletters,
   ]
   ```
4. If the tool is read-only, add its name to `READ_ONLY_TOOL_NAMES` in `backend/app/rag/graph_agent.py`.

---

### 5.2 How to Create a New Database Migration
1. Modify or add models in `backend/app/models/`.
2. Generate an Alembic migration from `backend/`:
   ```bash
   alembic revision -m "add_my_new_column"
   ```
3. Open the newly generated script in `backend/alembic/versions/` and define the `upgrade()` and `downgrade()` functions.
4. Apply the migration:
   ```bash
   alembic upgrade head
   ```

---

### 5.3 How to Add a New API Endpoint
1. Open the appropriate router in `backend/app/api/routes/` (or create a new router and register it in `backend/app/api/app.py`).
2. Define the route handler with dependency injection:
   ```python
   @router.get("/my-endpoint")
   def get_my_data(
       user: User = Depends(get_current_user),
       db: Session = Depends(get_db),
   ):
       # Database session automatically targets central or personal DB
       ...
       return {"status": "ok"}
   ```
3. Expose the endpoint function in `frontend/src/services/api.js`:
   ```javascript
   export const fetchMyData = () => api.get('/api/my-endpoint');
   ```

---

## 6. Testing & Verification

Run the automated test suite from the repository root:
```bash
pytest
```

To verify database schema completeness against all models:
```bash
python -c "from app.database.connection import verify_schema; verify_schema()"
```
This inspects the connected database and asserts that all 13 required tables and columns are present.

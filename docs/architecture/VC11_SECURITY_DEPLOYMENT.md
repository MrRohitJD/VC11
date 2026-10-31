# VC11 — Complete Security & Deployment Architecture

> **Verification:** Verified directly against `backend/app/core/security.py`, `backend/app/services/developer_settings.py`, `backend/app/services/chat_quota.py`, `Dockerfile`, `heroku.yml`, `DEPLOY_HEROKU.md`, and backend configurations.

---

## Table of Contents

1. [Security Architecture & Trust Boundaries](#1-security-architecture--trust-boundaries)
2. [Authentication & Session Security](#2-authentication--session-security)
3. [Authorization & Multi-Tenant User Isolation](#3-authorization--multi-tenant-user-isolation)
4. [Developer Personal Database Security (SSRF & Credential Protection)](#4-developer-personal-database-security-ssrf--credential-protection)
5. [OAuth Token Protection & Storage](#5-oauth-token-protection--storage)
6. [Prompt Injection, Input Sanitization & Redaction](#6-prompt-injection-input-sanitization--redaction)
7. [Rate Limiting, Quotas & Concurrency Locks](#7-rate-limiting-quotas--concurrency-locks)
8. [Security Controls Evaluation: Implemented vs. Gaps](#8-security-controls-evaluation-implemented-vs-gaps)
9. [Deployment Architecture](#9-deployment-architecture)
   - [9.1 Containerization & Multi-Stage Dockerfile](#91-containerization--multi-stage-dockerfile)
   - [9.2 Heroku PaaS Configuration & Release Lifecycle](#92-heroku-paas-configuration--release-lifecycle)
   - [9.3 Production Topology & External Integrations](#93-production-topology--external-integrations)
   - [9.4 Environment Variables Reference (No Secrets)](#94-environment-variables-reference-no-secrets)
   - [9.5 Production Verification & Health Checks](#95-production-verification--health-checks)

---

## 1. Security Architecture & Trust Boundaries

```mermaid
flowchart TB
    subgraph PUBLIC["🌐 Public Internet (Untrusted)"]
        CLIENT[User Web Browser]
        ATTACKER[Malicious Actor / Scanner]
    end

    subgraph EDGE["🛡️ Edge & Ingress Layer (Heroku / HTTPS)"]
        SSL[TLS 1.3 Termination\nCustom Domain: mail-assistant.rohit-jadhav.dev]
        CORS[CORSMiddleware\nAllow-Origin: frontend_url only]
        DOM_REDIR[Domain Redirect Middleware]
    end

    subgraph CONTAINER["📦 Application Container Boundary (FastAPI)"]
        DEMO_WALL[protect_demo_mailbox Middleware\nBlocks all mutating calls for demo user]
        DEV_ROUTER[route_developer_connections Middleware\nPer-request personal DB context]
        AUTH_GUARD[HMAC Session Verification\napp/core/security.py]
        
        subgraph APP_LOGIC["Backend Application Services"]
            ROUTERS[FastAPI Routers]
            RAG_AGENT[LangGraph Agent & Tools]
            QUOTA_GUARD[Chat Quota & Concurrency Leases]
            SANITIZER[DOMPurify & BeautifulSoup Preprocessors]
        end
    end

    subgraph DATA_BOUNDARY["🗄️ Persistence Layer (Isolated)"]
        CENTRAL_DB[(Heroku Postgres + pgvector\nRow-Level User Isolation)]
        ENC_STORE[Fernet AES Encryption at Rest\nPersonal DB URLs & LLM Keys]
    end

    subgraph USER_EXTERNAL["🔌 User-Supplied Infrastructure (Untrusted)"]
        PINNED_IP[SSRF Guard: DNS Hostaddr Pinning\n5s Connect Timeout & NullPool]
        USER_DB[(User Personal PostgreSQL)]
    end

    subgraph EXT_PROVIDERS["☁️ External Third-Party APIs"]
        GOOGLE_API[Google OAuth 2.0 & Gmail API v1]
        AZURE_API[Azure OpenAI Service]
        LANGSMITH_API[LangSmith Observability]
    end

    CLIENT -->|HTTPS / WSS| SSL
    ATTACKER -.->|Blocked| SSL
    SSL --> CORS --> DOM_REDIR --> DEMO_WALL --> DEV_ROUTER --> AUTH_GUARD
    AUTH_GUARD --> APP_LOGIC
    
    APP_LOGIC --> QUOTA_GUARD
    APP_LOGIC --> CENTRAL_DB
    APP_LOGIC --> ENC_STORE
    
    DEV_ROUTER --> PINNED_IP --> USER_DB
    
    APP_LOGIC --> GOOGLE_API
    APP_LOGIC --> AZURE_API
    APP_LOGIC --> LANGSMITH_API
```

---

## 2. Authentication & Session Security

- **Session Architecture:** Uses a stateless, HMAC-SHA256 signed session cookie named `gmail_ai_session`.
- **Payload Structure:** `{"user_id": "<uuid>", "exp": <timestamp>}` encoded as URL-safe base64.
- **Tamper Protection:** The signature is generated via `hmac.new(settings.session_secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()`. Any modification of user ID or expiration invalidates the signature, yielding an immediate `401 Unauthorized`.
- **Cookie Security Flags:**
  - `HttpOnly = True`: JavaScript cannot read or extract the session cookie via XSS.
  - `SameSite = "lax"`: Prevents cross-site request forgery in typical third-party navigation.
  - `Secure`: Enforced in production (`settings.secure_cookies = True`).
- **Session Durations:**
  - Regular Users: 7 days (`604,800` seconds).
  - Demo Users: 24 hours (`86,400` seconds).

---

## 3. Authorization & Multi-Tenant User Isolation

VC11 enforces multi-tenancy at the data layer using strict foreign key relationships and mandatory join constraints:
- **No Shared Mailbox Access:** All queries retrieving emails, labels, categories, chunks, embeddings, or chat sessions join the authenticated user's `GmailAccount.user_id`:
  ```python
  stmt = (
      select(Email)
      .join(GmailAccount, GmailAccount.id == Email.gmail_account_id)
      .where(Email.id == email_id, GmailAccount.user_id == user.id)
  )
  ```
- **Vector Search Isolation:** When executing pgvector similarity searches, the vector query joins `EmailChunk` and `Email` back to `GmailAccount.user_id == user.id`. A user can never search or retrieve embeddings belonging to another account.
- **Demo Mode Isolation:** Demo users (`is_demo = True`) are blocked at the middleware level (`protect_demo_mailbox`) from executing any `POST`, `PUT`, `PATCH`, or `DELETE` requests across `/api/` endpoints (with safe exceptions for demo login and demo send simulation). Outbound email sending and live Gmail MCP discovery are strictly disabled.

---

## 4. Developer Personal Database Security (SSRF & Credential Protection)

The Developer Settings feature allows users to provide their own PostgreSQL database connection strings (`developer_database_url`). To prevent attackers from using this feature to probe internal cloud networks (Server-Side Request Forgery - SSRF):

1. **DNS Pinning & SSRF Mitigation (`app/services/developer_settings.py:get_pinned_hostaddr`):**
   - Resolves the hostname to an IP address at connection initiation.
   - Rejects private, loopback, link-local, and reserved IP ranges (e.g., `127.0.0.1`, `10.0.0.0/8`, `192.168.0.0/16`, `169.254.169.254` AWS metadata) unless `DEVELOPER_DATABASE_ALLOW_PRIVATE_NETWORKING` is explicitly enabled.
   - Passes the resolved IP via `hostaddr` in `connect_args`, preventing **Time-of-Check to Time-of-Use (TOCTOU) DNS rebinding attacks**.
2. **Short Connection Timeout:** Configured with `connect_timeout = 5` seconds to prevent denial of service via hung sockets.
3. **Connection Isolation:** Uses SQLAlchemy `NullPool` so connections to user-supplied databases are closed immediately and never pooled or shared across worker threads.
4. **Encryption at Rest:** Personal database URLs and custom LLM API keys are encrypted at rest in the `users` table using Fernet symmetric encryption (`DEVELOPER_SETTINGS_ENCRYPTION_KEY`).
5. **No Credential Leakage:** API endpoints returning developer settings mask passwords (`postgresql://user:***@host:5432/db`) and never return stored API keys in plaintext.

---

## 5. OAuth Token Protection & Storage

- **PKCE & State Validation:** The OAuth flow creates a cryptographic `state` parameter and PKCE `code_verifier` stored in short-lived (10-minute) `HttpOnly` cookies. Upon callback, the state and verifier are strictly validated before exchanging tokens.
- **Token Verification:** The Google `id_token` is cryptographically validated using Google's public keys (`id_token.verify_oauth2_token`) and asserts that `email_verified` is true.
- **Automatic Token Refresh:** When communicating with the Gmail API, the Google client library automatically handles refreshing expired access tokens using the stored `refresh_token`.

---

## 6. Prompt Injection, Input Sanitization & Redaction

- **Tool Argument Redaction (`app/rag/tools.py:sanitize_tool_args`):** Before logging or returning tool call arguments in telemetry, sensitive fields (`body`, `draft_body`, `token`, `access_token`, `refresh_token`, `password`, `secret`) are automatically redacted to `[REDACTED]`.
- **Email HTML Sanitization:**
  - **Backend:** Raw email HTML is parsed and stripped of executable scripts, stylesheets, iframes, and active objects via BeautifulSoup in `clean_email_text()` before text chunking and embedding.
  - **Frontend:** Rendered bodies in `EmailPreview` and `EmailDetailsModal` are purified via `DOMPurify.sanitize()` prior to DOM injection.
- **Prompt Injection Defense:** System prompts for the intent router, planner, and agent explicitly separate user instructions from retrieved email context, instructing the model to treat retrieved content as unverified data rather than instructions.

---

## 7. Rate Limiting, Quotas & Concurrency Locks

*Source file:* `backend/app/services/chat_quota.py`

VC11 incorporates an atomic, database-backed quota and concurrency control engine:

| User Type | Daily Chat Quota | Concurrent Request Limit | Quota Key |
|---|---|---|---|
| **Regular User** | 50 requests / day | 5 concurrent requests | `user:{uuid}` |
| **Developer (Custom LLM Key)** | 500 requests / day | 5 concurrent requests | `user:{uuid}` |
| **Demo Visitor** | 10 requests / day | 4 concurrent requests | `demo:{visitor_hash}` |

### Reservation Algorithm (`reserve_chat_request`):
1. Acquires a row-level lock (`SELECT ... FOR UPDATE`) on the identity's `chat_concurrency_locks` row.
2. Purges expired leases older than 3 minutes.
3. Checks active leases against the concurrency limit; if exceeded, raises `429 Too Many Requests`.
4. Checks today's usage in `chat_quotas`; if exceeded, raises `429 Too Many Requests` with a `Retry-After` header.
5. Increments daily count and issues a `ChatConcurrencyLease` with a background heartbeat thread extending the lease every 30 seconds until the request completes.

---

## 8. Security Controls Evaluation: Implemented vs. Gaps

| Security Control | Implementation Mechanism | Status | Notes |
|---|---|---|---|
| **Authentication** | HMAC-SHA256 signed cookie | ✅ IMPLEMENTED | High entropy session secret |
| **Authorization / Tenant Isolation**| Joined user ID conditions | ✅ IMPLEMENTED | Strict multi-tenancy |
| **OAuth 2.0 PKCE** | Code verifier cookie | ✅ IMPLEMENTED | Mitigates authorization code injection |
| **Personal DB SSRF Protection** | DNS pinning via hostaddr + NullPool | ✅ IMPLEMENTED | Blocks AWS metadata & private LANs |
| **Credential Encryption at Rest** | Fernet AES encryption in DB | ✅ IMPLEMENTED | Encrypts DB URLs & LLM keys |
| **Human Approval for Outbound Mail**| `EmailSendApproval` PENDING gate | ✅ IMPLEMENTED | Agent cannot send without user click |
| **Chat Quotas & Concurrency** | Row-locked DB leases | ✅ IMPLEMENTED | Protects LLM budget & server threads |
| **XSS Prevention in HTML Emails** | DOMPurify on frontend | ✅ IMPLEMENTED | Sanitizes HTML before rendering |
| **SQL Injection Prevention** | SQLAlchemy parameterized queries | ✅ IMPLEMENTED | Zero raw string interpolation queries |
| **API Rate Limiting (General)** | General endpoint protection | ⚠️ GAP / PLANNED | Quota only applies to AI chat |
| **CSRF Anti-Forgery Token** | Explicit token header | ⚠️ PARTIAL | Relies on SameSite=Lax cookies |
| **OAuth Token DB Encryption** | Google refresh tokens encrypted at rest | ⚠️ GAP / PLANNED | Stored in DB; should use Fernet |

---

## 9. Deployment Architecture

### 9.1 Containerization & Multi-Stage Dockerfile
*File:* `Dockerfile`

VC11 is packaged as a unified container containing both the React frontend and FastAPI backend:
1. **Stage 1 (Node 22 Alpine):** Builds Vite React SPA assets into `/build/dist`.
2. **Stage 2 (Python 3.12 Slim):** Installs backend Python dependencies from `requirements-heroku.txt`, copies backend code and compiled frontend assets into `/app/frontend/dist`.
3. **Execution:** Runs `uvicorn app.api.app:app --host 0.0.0.0 --port ${PORT:-8000}`.

### 9.2 Heroku PaaS Configuration & Release Lifecycle
*Files:* `heroku.yml`, `DEPLOY_HEROKU.md`

- **Stack:** Container stack (`heroku.yml`).
- **Release Phase:** Before rolling out the new dyno, Heroku executes:
  ```bash
  sh scripts/heroku_release.sh
  ```
  This automatically runs:
  1. `alembic upgrade head`: Applies any pending PostgreSQL and pgvector schema migrations.
  2. `python scripts/seed_demo_mailbox.py`: Seeds the read-only demo account idempotently.
- **Web Dyno:** Starts Uvicorn ASGI server serving API endpoints and static SPA fallback.

### 9.3 Production Topology & External Integrations

```mermaid
flowchart LR
    subgraph HEROKU["Heroku Cloud Platform"]
        DYNO[Web Dyno: Container\nUvicorn + FastAPI + React SPA]
        PG[(Heroku Postgres\npgvector add-on\nHALFVEC 3072)]
    end

    subgraph GOOGLE["Google Cloud Platform"]
        G_OAUTH[Google Identity OAuth 2.0]
        G_MAIL[Gmail API v1]
    end

    subgraph AZURE["Azure Cognitive Services"]
        AZ_CHAT[Azure OpenAI: gpt-4.1-mini]
        AZ_EMBED[Azure OpenAI: text-embedding-3-large]
    end

    subgraph MONITOR["Observability"]
        L_SMITH[LangSmith Tracing]
    end

    USER[End User Browser] -->|HTTPS| DYNO
    DYNO -->|psycopg pooled| PG
    DYNO --> G_OAUTH & G_MAIL
    DYNO --> AZ_CHAT & AZ_EMBED
    DYNO -.-> L_SMITH
```

### 9.4 Environment Variables Reference (No Secrets)

| Variable Name | Required | Default | Purpose |
|---|---|---|---|
| `ENVIRONMENT` | Yes | `production` | Enables strict security checks |
| `DATABASE_URL` | Yes | — | PostgreSQL connection string (Heroku add-on) |
| `RAG_VECTOR_BACKEND` | Yes | `postgres` | Sets PostgreSQL pgvector as vector store |
| `DATABASE_POOL_SIZE` | No | `5` | SQLAlchemy pool size |
| `DATABASE_MAX_OVERFLOW`| No | `10` | SQLAlchemy maximum pool overflow |
| `SESSION_SECRET` | Yes | — | HMAC key for session cookies (min 32 chars) |
| `DEVELOPER_SETTINGS_ENCRYPTION_KEY` | Yes | — | Fernet key for encrypting developer DB URLs |
| `SECURE_COOKIES` | Yes | `true` | Enforces `Secure` flag on cookies |
| `FRONTEND_URL` | Yes | `https://mail-assistant.rohit-jadhav.dev` | Allowed CORS origin & redirect base |
| `BACKEND_URL` | Yes | `https://mail-assistant.rohit-jadhav.dev` | Base URL for API |
| `OAUTH_GOOGLE_REDIRECT_URI` | Yes | `https://mail-assistant.rohit-jadhav.dev/api/auth/callback` | OAuth callback URI |
| `GOOGLE_CLIENT_ID` | Yes | — | Google Cloud OAuth Client ID |
| `GOOGLE_CLIENT_SECRET` | Yes | — | Google Cloud OAuth Client Secret |
| `AZURE_OPENAI_ENDPOINT` | Yes | — | Azure OpenAI base URL |
| `AZURE_OPENAI_API_KEY` | Yes | — | Azure OpenAI access key |
| `AZURE_OPENAI_DEPLOYMENT`| Yes | `gpt-4.1-mini` | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Yes | `text-embedding-3-large`| Embedding model deployment name |
| `LANGSMITH_TRACING` | No | `false` | Enables LangSmith telemetry |
| `LANGSMITH_API_KEY` | No | — | LangSmith authentication key |
| `LANGSMITH_PROJECT` | No | `mailbox-devLG` | LangSmith project workspace |

### 9.5 Production Verification & Health Checks
- **Health Check:** `GET /health` returns `{"status": "ok"}` with HTTP 200.
- **Database Verification:** `init_db()` runs `verify_schema()` at startup, failing loudly if any of the 13 required tables are missing from Alembic migrations.

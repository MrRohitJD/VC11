# Gmail AI Dashboard & Inbox Assistant (VC11)

> **Enterprise-grade, AI-powered personal intelligence hub for Gmail.**  
> Grounded question-answering, multi-agent workflows (LangGraph), pgvector semantic search (`HALFVEC(3072)`), live Model Context Protocol (MCP) tool execution, and dual-mode PostgreSQL persistence.

---

## 🌟 Overview

**VC11** transforms a standard Gmail inbox into a grounded, private AI assistant. It synchronizes emails into PostgreSQL, classifies messages into AI categories, vectors chunks using Azure OpenAI's `text-embedding-3-large` into `pgvector`, and powers an intelligent assistant using **LangGraph** with 10 production agent patterns.

Developers and users can optionally attach their own **Personal PostgreSQL Database** and custom **LLM API Keys** (Google Gemini or OpenAI) via the Developer Settings panel, isolating their email data with strict zero-fallback boundaries.

---

## 🚀 Key Features

* **Grounded RAG Assistant:** Question answering backed by source citations (`[1]`, `[2]`), with an interactive citation inspector and zero-hallucination guardrails.
* **LangGraph Multi-Agent Orchestration:** 10 production agent patterns (Router, Tool-Calling, ReAct, Plan-and-Execute, Contextual Compression, Hybrid Retrieval, etc.) with state checkpointing.
* **pgvector Vector Database:** Uses 3072-dimensional half-precision vectors (`HALFVEC(3072)`) for high-performance semantic retrieval.
* **Dual-Engine Hybrid Search:** Fuses dense vector cosine similarity with sparse BM25 lexical keyword matching (with $3\times$ subject boosting) using **Reciprocal Rank Fusion (RRF)**.
* **Model Context Protocol (MCP) Server:** Built-in read-only Gmail MCP service (`search_mailbox`) over standard I/O and in-process RPC.
* **Human-in-the-Loop Safe Email Delivery:** Autonomous drafting creates `PENDING` approvals; emails are never sent without explicit human review and authorization.
* **Dual-Mode Persistence (Application DB vs. Personal DB):** Run against the central deployment database or connect an external personal PostgreSQL instance with strict failure isolation (never silent fallback).
* **Real-time Telemetry (SSE):** Live stopwatch, progress bars, and observable tool activity lanes streamed to the frontend via Server-Sent Events.
* **Interactive 3D Visualizer:** Includes a standalone WebGL 3D architecture visualizer with OrbitControls and a component inspector.

---

## 🏛️ System Architecture

```mermaid
flowchart TB
    subgraph CLIENT["Frontend (Vite + React 18)"]
        UI[App.jsx / AiAssistant.jsx / SyncPage.jsx]
        API_SVC[services/api.js - Axios + SSE Stream]
    end

    subgraph API_GATEWAY["Backend Gateway (FastAPI)"]
        CORS[CORSMiddleware]
        DEV_CTX[route_developer_connections Middleware]
        DEMO_WALL[protect_demo_mailbox Middleware]
        AUTH[HMAC-SHA256 Session Guard]
    end

    subgraph BACKEND_SERVICES["FastAPI Backend Services"]
        R_AUTH[/api/auth]
        R_EMAILS[/api/emails]
        R_GMAIL[/api/gmail]
        R_LLM[/api/llm]
        R_RAG[/api/rag]
        R_DEV[/api/developer]
    end

    subgraph AI_CORE["AI, LangGraph & Retrieval"]
        GRAPH[LangGraph StateGraph Agent]
        RETRIEVER[GmailHybridRetriever: pgvector + BM25]
        TOOLS[18 LangChain Tools]
        MCP[Gmail MCP Server]
    end

    subgraph PERSISTENCE["Dual-Mode PostgreSQL Storage"]
        DB_SWITCH{"ContextVar Switch"}
        APP_DB[(Central Application DB\nPostgreSQL + pgvector)]
        PERS_DB[(User Personal DB\nStrict Failure Mode)]
    end

    subgraph EXTERNAL["Cloud Services"]
        G_OAUTH[Google OAuth 2.0 PKCE]
        G_API[Gmail REST API v1]
        AZ_OAI[Azure OpenAI: gpt-4.1-mini & text-embedding-3-large]
        L_SMITH[LangSmith Observability]
    end

    CLIENT --> API_GATEWAY --> BACKEND_SERVICES
    R_AUTH --> G_OAUTH
    R_GMAIL --> G_API
    R_RAG --> GRAPH
    GRAPH --> RETRIEVER & TOOLS & AZ_OAI & L_SMITH
    TOOLS --> MCP --> G_API
    RETRIEVER --> PERSISTENCE
    BACKEND_SERVICES --> DB_SWITCH
    DB_SWITCH -->|Default| APP_DB
    DB_SWITCH -->|Personal URL set| PERS_DB
```

---

## 📁 Repository Directory Structure

```text
d:\Project\VC11/
├── Dockerfile                              # Multi-stage production container build (Node 22 + Python 3.12)
├── heroku.yml                              # Heroku PaaS container stack & release phase config
├── DEPLOY_HEROKU.md                        # Production deployment guide
├── docs/
│   └── architecture/                       # 📖 COMPLETE ARCHITECTURE DOCUMENTATION SUITE
│       ├── VC11_MASTER_ARCHITECTURE.md     # Master high-level architecture & system flows
│       ├── VC11_LOW_LEVEL_ARCHITECTURE.md  # Detailed LLD: classes, algorithms, state machines, SQL
│       ├── VC11_DATABASE_ERD.md            # Table-level ERDs, pgvector schemas, migration index
│       ├── VC11_API_CATALOG.md             # Complete REST & SSE endpoint catalogue
│       ├── VC11_FRONTEND_INTERACTIONS.md   # Route maps & 30-row Button-to-API Master Table
│       ├── VC11_AI_RAG_MCP.md              # AI, RAG, 10 agent patterns, 18 tools & MCP service
│       ├── VC11_SECURITY_DEPLOYMENT.md     # SSRF defense, Fernet encryption & cloud topology
│       ├── VC11_GLOSSARY.md                # System terminology index
│       ├── DEVELOPER_GUIDE.md              # Developer onboarding & code navigation handbook
│       └── interactive_architecture_viewer.html # 🎮 Interactive 3D WebGL Architecture Map
├── gmail-ai-dashboard/
│   ├── backend/
│   │   ├── alembic/                        # Database migration scripts (16 versioned migrations)
│   │   ├── app/
│   │   │   ├── api/                        # FastAPI entrypoint (app.py) and route definitions
│   │   │   ├── core/                       # Settings (config.py) and security (security.py)
│   │   │   ├── database/                   # Connection management, SessionLocal, and CRUD
│   │   │   ├── models/                     # SQLAlchemy declarative models (email, rag, llm, quota)
│   │   │   ├── rag/                        # LangGraph graph agent, tools, hybrid retrieval, BM25
│   │   │   └── services/                   # Business logic (Gmail, RAG indexer, MCP, scheduler)
│   │   └── requirements.txt                # Python backend dependencies
│   └── frontend/
│       ├── src/
│       │   ├── components/                 # Modals, Compose window, SearchActivity, Sent view
│       │   ├── context/                    # AuthContext (global session management)
│       │   ├── pages/                      # Dashboard, AiAssistant, SyncPage, GmailConsent
│       │   ├── services/                   # api.js (Axios + SSE TextDecoder reader)
│       │   └── App.jsx                     # Route definitions and dashboard views
│       └── package.json                    # React 18 & Vite frontend dependencies
└── tests/                                  # Integration and unit test suite
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React 18, Vite, React Router v6, Recharts, Lucide React, DOMPurify |
| **Backend** | Python 3.12+, FastAPI, Uvicorn, SQLAlchemy 2.x, Alembic, Pydantic v2 |
| **Database** | PostgreSQL 16+, `pgvector` (`HALFVEC(3072)`), SQLite (local checkpointer) |
| **AI / Orchestration** | LangGraph (`StateGraph`), LangChain Core, LangChain Community |
| **LLM Inference** | Azure OpenAI (`gpt-4.1-mini`), Google Gemini (`gemini-2.5-flash`), OpenAI |
| **Embeddings** | Azure OpenAI (`text-embedding-3-large`, 3072 dimensions) |
| **Protocols & Standards**| Model Context Protocol (MCP), Server-Sent Events (SSE), OAuth 2.0 PKCE |
| **Observability** | LangSmith Tracing, Custom Real-Time Activity Streams |
| **Deployment** | Docker, Heroku Container Stack |

---

## ⚡ Quick Start & Local Setup

### Prerequisites
- Python 3.12+
- Node.js 20+ & npm
- PostgreSQL 16+ with the `pgvector` extension enabled (`CREATE EXTENSION IF NOT EXISTS vector;`)
- Google Cloud OAuth 2.0 Client Credentials (with `gmail.readonly` and `gmail.send` scopes)

### 1. Database Setup
```bash
# Connect to your PostgreSQL instance and create the database
psql -U postgres -c "CREATE DATABASE gmail_ai_dashboard;"
psql -U postgres -d gmail_ai_dashboard -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 2. Backend Setup
```bash
cd gmail-ai-dashboard/backend

# Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and supply your DATABASE_URL, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, AZURE_OPENAI_API_KEY, etc.

# Run database migrations
alembic upgrade head

# Start the FastAPI server
uvicorn app.api.app:app --reload --port 8000
```
Backend API will be running at `http://localhost:8000`.

### 3. Frontend Setup
```bash
cd ../frontend

# Install npm packages
npm install

# Start Vite dev server
npm run dev
```
Frontend web application will be accessible at `http://localhost:5173`.

---

## 📖 Deep-Dive Architecture Documentation

For complete technical specifications, review the dedicated guides in [`docs/architecture/`](file:///d:/Project/VC11/docs/architecture/):

1. [**Master Architecture (`VC11_MASTER_ARCHITECTURE.md`)**](file:///d:/Project/VC11/docs/architecture/VC11_MASTER_ARCHITECTURE.md): Complete system topology, 14 architectural zones, and full end-to-end sequence diagrams.
2. [**Low-Level Architecture (`VC11_LOW_LEVEL_ARCHITECTURE.md`)**](file:///d:/Project/VC11/docs/architecture/VC11_LOW_LEVEL_ARCHITECTURE.md): Code-level design, class hierarchies, state machine transitions, BM25/RRF math, SSE wire frames, and row-level SQL locks.
3. [**Database ERD & Migrations (`VC11_DATABASE_ERD.md`)**](file:///d:/Project/VC11/docs/architecture/VC11_DATABASE_ERD.md): Complete entity relationship diagrams, pgvector schema, and 16-migration history.
4. [**API Catalogue (`VC11_API_CATALOG.md`)**](file:///d:/Project/VC11/docs/architecture/VC11_API_CATALOG.md): Every REST and SSE endpoint documented with request parameters, schemas, tables, and callers.
5. [**Frontend Interactions Map (`VC11_FRONTEND_INTERACTIONS.md`)**](file:///d:/Project/VC11/docs/architecture/VC11_FRONTEND_INTERACTIONS.md): Component navigation tree and the 30-row Button-to-API Master Table.
6. [**AI, RAG & MCP Architecture (`VC11_AI_RAG_MCP.md`)**](file:///d:/Project/VC11/docs/architecture/VC11_AI_RAG_MCP.md): Ingestion pipeline, 7 search mechanisms, 10 agent patterns, all 18 tools, and MCP server.
7. [**Security & Deployment (`VC11_SECURITY_DEPLOYMENT.md`)**](file:///d:/Project/VC11/docs/architecture/VC11_SECURITY_DEPLOYMENT.md): Personal DB SSRF mitigation (DNS hostaddr pinning), Fernet encryption at rest, chat quotas, and Heroku deployment.
8. [**Glossary & Index (`VC11_GLOSSARY.md`)**](file:///d:/Project/VC11/docs/architecture/VC11_GLOSSARY.md): Definitions for all architectural, AI, and database terms.
9. [**Developer Handbook (`DEVELOPER_GUIDE.md`)**](file:///d:/Project/VC11/docs/architecture/DEVELOPER_GUIDE.md): Code navigation, debugging, testing, and extension guide.
10. [**3D Interactive Architecture Map (`interactive_architecture_viewer.html`)**](file:///d:/Project/VC11/docs/architecture/interactive_architecture_viewer.html): Interactive 3D WebGL architecture visualizer.

---

## 🔒 Security & Privacy

* **Strict Isolation:** User emails, vector embeddings, and chat histories are strictly isolated via tenant foreign keys.
* **SSRF Defense:** User-supplied database URLs are checked against private IP ranges and pinned at DNS resolution (`hostaddr`).
* **Encryption at Rest:** Saved database URLs and custom API keys are encrypted at rest using Fernet symmetric encryption.
* **Human-in-the-Loop:** Outbound emails created by the AI assistant are held in `PENDING` approval status until confirmed by the user.
* **XSS Sanitization:** Email bodies are sanitized via `DOMPurify` before DOM injection.

---

## 📄 License & Attribution

Internal project codebase for **Gmail AI Dashboard (VC11)**.  
Built with FastAPI, React, LangGraph, and PostgreSQL pgvector.

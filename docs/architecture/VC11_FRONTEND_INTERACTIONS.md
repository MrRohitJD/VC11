# VC11 — Complete Frontend Interactions & Button-to-API Master Map

> **Verification:** Verified directly against `frontend/src/App.jsx`, `frontend/src/pages/AiAssistant.jsx`, `frontend/src/pages/SyncPage.jsx`, `frontend/src/pages/GmailConsent.jsx`, `frontend/src/pages/PublicPages.jsx`, `frontend/src/components/*`, and `frontend/src/services/api.js`.

---

## Table of Contents

1. [Frontend Architecture & State Overview](#1-frontend-architecture--state-overview)
2. [Route Navigation Map](#2-route-navigation-map)
3. [Page-by-Page Technical Specifications](#3-page-by-page-technical-specifications)
   - [3.1 Public Home, Privacy Policy, Terms (`/`, `/privacy`, `/terms`)](#31-public-home-privacy-policy-terms---privacy--terms)
   - [3.2 Authentication (`/login`, `/signup`)](#32-authentication--login--signup)
   - [3.3 Gmail Consent (`/consent`)](#33-gmail-consent--consent)
   - [3.4 Mailbox Sync (`/sync`)](#34-mailbox-sync--sync)
   - [3.5 Main Dashboard (`/dashboard`)](#35-main-dashboard--dashboard)
   - [3.6 AI Inbox Assistant (`/assistant`)](#36-ai-inbox-assistant--assistant)
   - [3.7 Settings & Developer Configuration (`/settings`)](#37-settings--developer-configuration--settings)
4. [Modal Dialogs & Secondary Components](#4-modal-dialogs--secondary-components)
   - [4.1 ComposeModal](#41-composemodal)
   - [4.2 EmailDetailsModal](#42-emaildetailsmodal)
   - [4.3 SearchActivity](#43-searchactivity)
   - [4.4 SentMessagesView](#44-sentmessagesview)
5. [Button-to-API Master Table](#5-button-to-api-master-table)

---

## 1. Frontend Architecture & State Overview

The VC11 frontend is built with React 18, Vite, and React Router v6.
- **Global Authentication Context (`AuthContext.jsx`)**: Holds `user`, `loading`, `isAuthenticated`. On initial mount, it invokes `getCurrentUser()`.
- **API Client (`services/api.js`)**: Axios instance configured with `withCredentials: true` and an interceptor that automatically attaches the client's local timezone via `X-Timezone` header and `tz` query parameter. It also provides the SSE reader implementation for streaming AI responses (`sendRagChat`).
- **Sanitization**: Uses `DOMPurify` to clean HTML bodies before injecting them into the DOM.

---

## 2. Route Navigation Map

```mermaid
flowchart TD
    ROOT["/ (PublicHome)"]
    PRIVACY["/privacy (PrivacyPolicy)"]
    TERMS["/terms (TermsOfService)"]
    LOGIN["/login (AuthPage - Sign In)"]
    SIGNUP["/signup (AuthPage - Sign Up)"]
    CONSENT["/consent (GmailConsent)"]
    SYNC["/sync (SyncPage - Protected)"]
    DASHBOARD["/dashboard (Dashboard - Protected)"]
    ASSISTANT["/assistant (AiAssistant - Protected)"]
    SETTINGS["/settings (SettingsPage - Protected)"]

    ROOT -->|Click 'Get Started'| LOGIN
    ROOT -->|Click 'Privacy'| PRIVACY
    ROOT -->|Click 'Terms'| TERMS
    ROOT -->|Click 'Open Demo'| DASHBOARD

    LOGIN -->|Click 'Sign up'| SIGNUP
    LOGIN -->|Click 'Try demo'| DASHBOARD
    LOGIN -->|Click 'Sign in with Google'| CONSENT
    SIGNUP -->|Click 'Sign in with Google'| CONSENT

    CONSENT -->|Click 'Continue with Google'| GOOGLE_OAUTH[Google Accounts OAuth]
    GOOGLE_OAUTH -->|OAuth Callback 302| SYNC

    SYNC -->|Sync Completed 100%| DASHBOARD
    
    DASHBOARD -->|Nav to AI Assistant| ASSISTANT
    DASHBOARD -->|Nav to Settings| SETTINGS
    DASHBOARD -->|Click Compose| COMPOSE_MODAL[ComposeModal]
    DASHBOARD -->|Click Email Row| PREVIEW_PANEL[Email Preview / Details]

    ASSISTANT -->|Click 'Back to Dashboard'| DASHBOARD
    ASSISTANT -->|Click Settings icon| SETTINGS
    ASSISTANT -->|Click Citation [1]| CITED_MODAL[EmailDetailsModal]
    ASSISTANT -->|Click 'Review subject & body'| DRAFT_MODAL[Draft Review Modal]

    SETTINGS -->|Click 'Log out'| LOGIN
    SETTINGS -->|Click 'Erase all data'| ERASE_DIALOG[EraseDialog]
    ERASE_DIALOG -->|Confirm| LOGIN
```

---

## 3. Page-by-Page Technical Specifications

### 3.1 Public Home, Privacy Policy, Terms (`/`, `/privacy`, `/terms`)
- **Component:** `PublicPages.jsx` (`PublicHome`, `PrivacyPolicy`, `TermsOfService`)
- **Access:** Public.
- **User Actions:**
  - Click "Open Demo" → calls `enterDemo()`, invokes `checkAuth()`, navigates to `/dashboard`.
  - Click "Get started" → navigates to `/login`.
  - Click Privacy / Terms links → navigates to corresponding documentation pages.

### 3.2 Authentication (`/login`, `/signup`)
- **Component:** `AuthPage` in `App.jsx`
- **Access:** Public (redirects to `/dashboard` if already logged in).
- **User Actions:**
  - Click "Sign in with Google" / "Sign up with Google" → navigates to `/consent`.
  - Click "Try interactive demo" → triggers `enterDemo()` API call → sets session cookie → navigates to `/dashboard`.
  - Click switch between sign in and sign up.

### 3.3 Gmail Consent (`/consent`)
- **Component:** `GmailConsent.jsx`
- **Access:** Public.
- **User Actions:**
  - Click "Continue with Google" → calls `window.location.assign(getGoogleAuthUrl())` (`/api/auth/google`).

### 3.4 Mailbox Sync (`/sync`)
- **Component:** `SyncPage.jsx`
- **Access:** Protected (Authenticated users only).
- **Data Loaded on Mount:**
  - Initiates `startSync()` (`POST /api/gmail/sync`).
  - Connects to SSE stream `EventSource(getSyncStreamUrl())` (`GET /api/gmail/sync/stream`).
  - Sets up fallback 2000ms polling via `getSyncStatus()` (`GET /api/gmail/sync/status`).
- **UI Progress Stages:** `starting` → `authenticating` → `labels` → `listing` → `fetching` → `storing` → `completed`.
- **Completion Behavior:** Automatically redirects to `/dashboard` after 2.5 seconds.
- **Error Behavior:** Renders error message and "Try Again" button (reloads page).

### 3.5 Main Dashboard (`/dashboard`)
- **Component:** `Dashboard` in `App.jsx`
- **Access:** Protected.
- **Data Loaded on Mount & Filter Change:**
  - `fetchAnalyticsSummary(range)` → KPI counts (Total, Auto-labeled, Important, Unread) vs. previous period.
  - `fetchAnalyticsCharts(activityRange)` → Category donut chart & incoming/outgoing activity line chart.
  - `fetchEmails({ per_page: 50, page: 1, inbox_only: true })` → Recent inbox emails table.
  - `fetchEmails({ per_page: 1, page: 1, sent_only: true })` → Sent emails count badge.
  - Background Polling: `getSyncStatus()` called every 15 seconds to monitor background syncing.
- **Interactive Controls:**
  - Date range selector dropdown (Today, Yesterday, This week, Last week, This month, Last month, 6 months, 1 year, All time).
  - Activity chart range selector dropdown.
  - Category tabs (All, Primary, Social, Promotions, Updates, Forums).
  - Select all checkbox & per-row checkboxes for bulk operations.
  - Star / unstar email button.
  - Search input (`⌘ K`) filtering emails by keyword or sender.
  - Quick action cards: "Sync Gmail", "Train Model", "Manage Labels", "View Analytics".
  - Email row click: Opens sliding `EmailPreview` panel with full content and action buttons (Reply, Forward, Expand modal, Close).

### 3.6 AI Inbox Assistant (`/assistant`)
- **Component:** `AiAssistant.jsx`
- **Access:** Protected.
- **Data Loaded on Mount:**
  - `getRagStatus()` → Checks vector indexing status and API health.
  - `getRagSessions(recentLimit)` → Loads chat history sidebar list.
- **Interactive Controls:**
  - Composer multiline textarea: Auto-expanding height, Enter to submit (Shift+Enter for newline).
  - Quick prompt buttons: "Search", "Summarize", "Draft Reply", "Meetings".
  - Real-time Execution Feed: Displays `SearchActivity` collapsible component showing observable LangGraph steps.
  - Inline Citations: Clicking cited badges (e.g. `[1]`, `[2]`) or "View email" opens `EmailDetailsModal`.
  - Human-in-the-Loop Approval Card: Rendered when the agent drafts an email. Offers "Review subject & body", "Approve & send", and "Discard".
  - Draft Review Modal: Allows editing recipient, CC, BCC, subject, and body before sending.
  - Side Panel Toggle: Displays extracted Key Points, Related cited emails, and Recent chat sessions.
  - Focus Mode Toggle: Hides sidebars for distraction-free chat.

### 3.7 Settings & Developer Configuration (`/settings`)
- **Component:** `SettingsPage` in `App.jsx`
- **Access:** Protected.
- **Data Loaded on Mount:**
  - `getExportStats()` → Total emails, HTML bodies count, date ranges.
  - `getRagStatus()` → Vector counts and index status.
  - `getDeveloperSettings()` → Personal DB URL and LLM key configuration status.
- **Interactive Controls:**
  - Download schema button: Triggers download of `gmail-ai-dashboard-schema.sql`.
  - Personal Database: Input connection string, click "Connect Database" (validates schema), or "Disconnect & Use App Database".
  - Custom LLM Key: Select provider (Google Gemini or OpenAI), enter key, click "Save API key" or "Use app key".
  - Vector Indexing: Click "Start Embeddings" to index un-embedded chunks.
  - Export: Checkbox "Include full email bodies", buttons "Export as CSV" and "Export as JSON".
  - Import: File input to upload JSON/CSV email dumps.
  - Danger Zone: "Erase All Data" button opens confirmation modal `EraseDialog`.

---

## 4. Modal Dialogs & Secondary Components

### 4.1 ComposeModal (`components/ComposeModal.jsx`)
- **Trigger:** Click "Compose" button in sidebar or "Reply"/"Forward" in email preview.
- **Controls:**
  - Header: Minimize, Maximize toggle, Close (X).
  - Fields: `To` (with format validation), `Cc` toggle and field, `Bcc` toggle and field, `Subject`, multiline `Body`.
  - Footer: "Send" button (supports `Ctrl+Enter` shortcut), "Discard" button.
- **API Call:** `sendEmail({ to, cc, bcc, subject, body })` (`POST /api/emails/send`).

### 4.2 EmailDetailsModal (`components/EmailDetailsModal.jsx`)
- **Trigger:** Clicking any citation badge `[1]` in AI Assistant or "View email" button.
- **Data Loading:** Calls `getRagSourceEmail(email.id)` (`GET /api/rag/sources/{email_id}`).
- **Features:** Shows headers (From, To, Cc, Bcc, Date), rendered body (DOMPurified HTML or plaintext), and attachment list.

### 4.3 SearchActivity (`components/SearchActivity.jsx`)
- **Trigger:** Rendered inline above or below AI Assistant answers during and after query execution.
- **Data Source:** SSE `event: activity` frames emitted by `POST /api/rag/chat/stream`.
- **Visuals:** Shows live stopwatch elapsed timer, progress bar, tool lanes (Route, Plan, Search, Validate, Tool Agent, Compose), and link to private LangSmith trace (if enabled).

### 4.4 SentMessagesView (`components/SentMessagesView.jsx`)
- **Trigger:** Click "Sent" folder in sidebar.
- **Data Loading:** `fetchEmails({ sent_only: true, page, per_page: 10, search })`.
- **Features:** Infinite scroll with IntersectionObserver, category tabs, and selection checkboxes.

---

## 5. Button-to-API Master Table

| # | Button / Action | Location / Component | Event Handler | API Endpoint Called | Method | Backend Service | Resulting UI Update |
|---|---|---|---|---|---|---|---|
| 1 | "Sign in with Google" | `AuthPage` (`App.jsx`) | `form onSubmit` | `/api/auth/google` | GET | `auth.py:google_login` | Redirects to Google consent screen |
| 2 | "Continue with Google" | `GmailConsent.jsx` | `handleContinue` | `/api/auth/google` | GET | `auth.py:google_login` | Redirects to Google OAuth flow |
| 3 | "Try interactive demo" | `AuthPage` / `PublicHome` | `tryDemo` / `openDemo` | `/api/auth/demo` | POST | `auth.py:enterDemo` | Sets session cookie, navigates to `/dashboard` |
| 4 | "Log out" | Topbar Profile / Settings | `handleLogout` | `/api/auth/logout` | POST | `auth.py:logout` | Clears session cookie, navigates to `/login` |
| 5 | "Erase All Data" (Confirm) | `EraseDialog` (`App.jsx`) | `handleConfirmErase` | `/api/auth/erase` | DELETE | `auth.py:erase_all_data` | Erases mailbox data, clears session, navigates to `/login` |
| 6 | "Sync Gmail" | QuickActions (`App.jsx`) | `handleSync` | `/api/gmail/sync` | POST | `gmail.py:start_sync` | Starts sync thread, displays sync banner |
| 7 | "Train Model" | QuickActions (`App.jsx`) | `handleTrain` | `/api/llm/classify/backfill` | POST | `llm.py:start_backfill` | Starts background classification, displays confirmation toast |
| 8 | KPI Date Range Dropdown | `App.jsx` (Stats Section) | `onRangeChange` | `/api/analytics/summary` | GET | `emails.py:analytics_summary` | Re-renders KPI cards with new metrics |
| 9 | Activity Range Dropdown | `Charts.jsx` (`App.jsx`) | `onRangeChange` | `/api/analytics/charts` | GET | `emails.py:analytics_charts` | Updates LineChart data points |
| 10 | Category Tab (e.g. Primary) | `InboxView` (`App.jsx`) | `setCategoryTab` | `/api/emails` | GET | `emails.py:list_emails` | Filters email list by Gmail category |
| 11 | Star / Unstar Email | `RecentEmails` (`App.jsx`) | `onToggleStar` | Local state / DB sync | — | — | Updates star icon state |
| 12 | Select All Checkbox | `RecentEmails` / `InboxView` | `onToggleSelectAll` | Local state | — | — | Toggles selection state of all visible rows |
| 13 | Email Row Click | `RecentEmails` / `InboxView` | `onSelectEmail` | `/api/emails/{id}` | GET | `emails.py:get_email` | Opens `EmailPreview` pane with full HTML |
| 14 | "Send" (Compose Modal) | `ComposeModal.jsx` | `handleSend` | `/api/emails/send` | POST | `emails.py:send_email_endpoint` | Sends email via Gmail API, closes modal |
| 15 | "Send Message" (AI Chat) | `AiAssistant.jsx` | `handleSend` | `/api/rag/chat/stream` | POST (SSE) | `rag.py:stream_chat` | Streams execution activity and renders answer |
| 16 | Suggested Prompt Card | `AiAssistant.jsx` | `handleSend(prompt)` | `/api/rag/chat/stream` | POST (SSE) | `rag.py:stream_chat` | Submits suggested prompt to assistant |
| 17 | Citation Link `[1]` | `AiAssistant.jsx` | `handleOpenEmailModal` | `/api/rag/sources/{id}` | GET | `rag.py:get_source_email_details` | Opens `EmailDetailsModal` with cited source |
| 18 | "Review subject & body" | `AiAssistant.jsx` (Approval Card) | `openDraftReview` | Local state | — | — | Opens `DraftReview` edit modal |
| 19 | "Save changes" (Draft Review)| `AiAssistant.jsx` | `handleSaveDraft` | `/api/rag/approvals/{id}` | PATCH | `rag.py:update_email_draft` | Updates draft text in database |
| 20 | "Approve & send" | `AiAssistant.jsx` | `handleApprove` | `/api/rag/approvals/{id}/approve` | POST | `rag.py:approve_email_send` | Delivers email via Gmail API, marks `SENT` |
| 21 | "Discard draft" | `AiAssistant.jsx` | `handleReject` | `/api/rag/approvals/{id}/reject` | POST | `rag.py:reject_email_send` | Marks draft `REJECTED`, removes approval card |
| 22 | "Connect Database" | `SettingsPage` (`App.jsx`) | `saveDeveloperSettings`| `/api/developer` | PUT | `developer.py:update_developer_settings` | Tests schema, encrypts URL, updates DB mode |
| 23 | "Disconnect & Use App DB"| `SettingsPage` (`App.jsx`) | `saveDeveloperSettings`| `/api/developer` | PUT | `developer.py:update_developer_settings` | Clears personal DB, reverts to application DB |
| 24 | "Save API key" | `SettingsPage` (`App.jsx`) | `saveDeveloperSettings`| `/api/developer` | PUT | `developer.py:update_developer_settings` | Encrypts custom LLM key, increases daily quota |
| 25 | "Use app key" | `SettingsPage` (`App.jsx`) | `saveDeveloperSettings`| `/api/developer` | PUT | `developer.py:update_developer_settings` | Clears custom key, reverts to system default |
| 26 | "Start Embeddings" | `SettingsPage` (`App.jsx`) | `handleStartEmbeddings`| `/api/rag/index` | POST | `rag.py:trigger_rag_indexing` | Generates pgvector embeddings for all chunks |
| 27 | "Export as CSV" | `SettingsPage` (`App.jsx`) | `window.open` | `/api/export/csv` | GET | `export.py:export_csv` | Downloads CSV email file |
| 28 | "Export as JSON" | `SettingsPage` (`App.jsx`) | `window.open` | `/api/export/json` | GET | `export.py:export_json` | Downloads JSON email file |
| 29 | "Choose file to import" | `SettingsPage` (`App.jsx`) | `handleFileUpload` | `/api/export/import` | POST | `export.py:import_emails_file` | Parses CSV/JSON, inserts emails, refreshes inbox |
| 30 | "Download database schema"| `SettingsPage` (`App.jsx`) | Direct Link | `/api/developer/schema.sql`| GET | `developer.py:download_schema` | Downloads `gmail-ai-dashboard-schema.sql` |

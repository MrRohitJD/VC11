# devLG mailbox assistant

The existing `/rag/chat` API now uses `app.rag.graph_agent`.
The original assistant remains in `app.rag.agent` as the reference implementation
and supplies reusable retrieval, query rewriting, and model configuration.

Workflow: context → route → grounded retrieval / conversational response / model
→ tools → model. Tool execution is sequential and bounded. Requests combining
search and drafting use the model/tool cycle. No generated private reasoning is
requested or shown; steps are summaries of observable tool execution.

## Run

From the backend directory, install `pip install -r requirements.txt`, then run
`python -m uvicorn app.api.app:app --reload --port 8000`. Use the existing frontend startup instructions.
Existing Azure/OpenAI and Gmail settings continue to apply.

## Live Gmail MCP search

The backend starts its own read-only Gmail MCP server over stdio for mailbox
searches. It uses the signed-in application's user's stored Google OAuth
credentials; no separate MCP service or Codex Gmail plugin login is required.
The Gmail connector in a Codex chat is a separate connection used for diagnosis.
No send, delete, or label-modification tools are exposed by this project server.
The backend handles MCP startup automatically; use the same two backend/frontend
commands above. Install the updated requirements in the environment running the API.

Defaults (optional overrides in `backend/.env`):

```env
GMAIL_MCP_ENABLED=true
GMAIL_MCP_MAX_MESSAGES=1000
GMAIL_MCP_TIMEOUT_SECONDS=30
```

Assistant search includes all Gmail categories and archived mail. Spam, Trash and
Drafts are excluded. The local Gmail sync supplies indexed retrieval; broad,
date-scoped, sender-only, and freshness-sensitive queries also search live Gmail
through MCP. The agent has separate read-only tools for MCP and direct Gmail API
search. Live discovery paginates Gmail, reads full messages, and refreshes matching
local records and labels. These records keep the existing source-view and citation
endpoints working without requiring re-embedding on the current request.
If live discovery fails or reaches its cap, the answer explicitly warns that
coverage may be incomplete and continues with synced mail. The chat API accepts
`use_gmail_mcp` to enable or disable MCP discovery for an individual request;
when omitted, query-based routing applies.

"All" requests enumerate eligible database records rather than ranked chunks.
"Last N months" subtracts calendar months with day clamping in the client's
timezone. Date-only summaries skip semantic relevance classification because
membership is determined by receipt dates and labels. Topical queries still
require valid model decisions; malformed output now tries the next configured
provider as well as handling ordinary provider connection failures. No lexical
guess is silently substituted for an unavailable AI verdict.

## Conversation memory

LangGraph's SQLite checkpointer writes graph state after each graph step to
`backend/data/langgraph.sqlite` by default. Thread keys combine account and chat
session IDs. Existing SQL chat history supplies the latest 20 messages for query
rewriting and model context, including after restarting the backend. Request state
resets counters and transient tool messages at each turn. Database sessions and
tools are never checkpointed. This is thread-scoped memory; cross-conversation
personal preferences and automatic interrupted-run recovery are not implemented.
SQLite is intended for local development; use a shared Postgres checkpointer
before deploying multiple backend instances. Deleting a SQL chat currently does
not purge its checkpoints; keep the checkpoint file private and include it in
your data retention policy.

## LangSmith

Copy the settings in `backend/.env.example` into the existing backend `.env`.
Register at https://smith.langchain.com and create an API key. Set
`LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and your project name. Use the API
endpoint for your LangSmith region (the example uses US). Restart the backend.
Graph nodes, nested model calls, tools, timings, inputs, outputs, and failures
appear in the configured project. Traces can contain mailbox content.

### Frontend search activity

The chat page uses `POST /api/rag/chat/stream`. The response streams `activity`
events and one final `result` event. Observable steps include routing, retrieval
selection, live Gmail discovery, BM25 ranking, candidate counts, evidence checks,
one corrective search if needed, answer writing, and review. The panel shows
elapsed time since the request began. It never displays private model reasoning.
Failures are explicit warning events. Interrupted requests are not automatically
replayed, which avoids accidentally creating duplicate drafts.

Every answer returns `steps`, `search_plan`, `trace_id`, `trace_url`, and
`tracing_enabled`. When LangSmith is configured, the backend creates one root
trace containing the LangGraph/model/retriever runs and returns its private URL.
The UI's **Open LangSmith trace** link requires access to the LangSmith project.
API keys remain on the backend; no traces are published publicly. Local activity
continues to work when tracing is off or the trace URL cannot be resolved.
The existing non-streaming `/api/rag/chat` endpoint remains available.

### Retrieval and agent patterns

The graph now routes through context, a public execution plan, retrieval or a
bounded tool loop, and a final source/status review. Retrieval selects:

- Relational enumeration for exhaustive and date-only requests (no top-k loss).
- Structured lookup for explicit message/thread IDs.
- pgvector semantic similarity plus BM25 keywords, combined with reciprocal rank
  fusion for ordinary questions. BM25 boosts subjects and expands common mailbox
  topic terms, and works even when an email lacks a vector embedding.
- MMR for diverse-topic/overview requests.
- Multi-query expansion for compound analytical searches; `use_multi_query`
  defaults to automatic, with true/false as explicit overrides. Original account,
  Primary, and receipt-date constraints survive every expansion.
- One bounded corrective validation pass over lower ranked keyword candidates
  when the first candidate set contains no validated matches. Provider failure
  does not trigger guesses about relevance.

Relevance extraction is the evidence gate before answer generation. The final
prompt uses request-specific layouts for interviews, bills and daily digests,
source-number citations, event/receipt date distinctions and explicit uncertainty.
The review node checks completion status and source coverage; it is not an
independent guarantee that every generated sentence is correct.

## Drafts and delivery

Drafts create pending approval records and return their exact approval IDs.
The graph does not expose the legacy send tool because it only updates a database
status without Gmail delivery. Approving a draft records APPROVED and explicitly
reports that delivery is unavailable. Real Gmail send integration and resumable
LangGraph approval interrupts remain future work.

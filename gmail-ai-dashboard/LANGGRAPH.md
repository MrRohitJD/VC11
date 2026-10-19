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
`uvicorn app.main:app --reload`. Use the existing frontend startup instructions.
Existing Azure/OpenAI and Gmail settings continue to apply.

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

## Drafts and delivery

Drafts create pending approval records and return their exact approval IDs.
The graph does not expose the legacy send tool because it only updates a database
status without Gmail delivery. Approving a draft records APPROVED and explicitly
reports that delivery is unavailable. Real Gmail send integration and resumable
LangGraph approval interrupts remain future work.

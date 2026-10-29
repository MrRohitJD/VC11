"""Account-bound MCP client for live Gmail discovery before local retrieval."""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

from app.core.config import settings
logger = logging.getLogger(__name__)
from app.rag.search_plan import gmail_search_query


MAX_LIVE_GMAIL_SECONDS = 15
MAX_LIVE_GMAIL_MESSAGES = 100


async def _search(user_id, query, *, full_scan=False):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from sqlalchemy.engine import make_url

    database_url = make_url(settings.database_url)
    if database_url.drivername.startswith('sqlite') and database_url.database not in (None, '', ':memory:'):
        database_url = database_url.set(database=str(Path(database_url.database).resolve()))

    timeout_seconds = (settings.gmail_mcp_timeout_seconds if full_scan else
                       min(settings.gmail_mcp_timeout_seconds, MAX_LIVE_GMAIL_SECONDS))
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "app.services.gmail_mcp_server"],
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, "GMAIL_MCP_USER_ID": str(user_id),
             "DATABASE_URL": database_url.render_as_string(hide_password=False),
             "GMAIL_MCP_TIMEOUT_SECONDS": str(timeout_seconds)},
    )
    started = time.monotonic()
    async with asyncio.timeout(timeout_seconds):
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=timeout_seconds)) as session:
                await session.initialize()
                result = await session.call_tool("search_mailbox_emails", {
                    "query": gmail_search_query(query, include_topic_terms=not full_scan),
                    # Startup and handshake consume the SAME deadline as Gmail
                    # I/O; leave time to return committed partial results.
                    "time_budget_seconds": max(0, timeout_seconds - (time.monotonic() - started) - 2),
                    # Broad categorization uses a larger, still bounded scan.
                    # Search status reports when the configured cap is reached.
                    "max_messages": (settings.gmail_mcp_max_messages if full_scan else
                                     min(settings.gmail_mcp_max_messages, MAX_LIVE_GMAIL_MESSAGES)),
                })
                if result.isError:
                    raise RuntimeError("Live Gmail MCP search failed")
                payload = result.structuredContent
                if payload is None:
                    payload = json.loads(next(c.text for c in result.content if c.type == "text"))
                if not isinstance(payload.get("email_ids"), list) or type(payload.get("complete")) is not bool:
                    raise ValueError("Invalid Gmail MCP response")
                return payload


def search_live_mailbox(user_id, query, *, full_scan=False):
    timeout_seconds = (settings.gmail_mcp_timeout_seconds if full_scan else
                       min(settings.gmail_mcp_timeout_seconds, MAX_LIVE_GMAIL_SECONDS))
    started = time.monotonic()
    try:
        return asyncio.run(_search(user_id, query, full_scan=full_scan))
    except Exception as exc:
        logger.warning("Live Gmail MCP search via subprocess failed (%s); trying direct Gmail service search", type(exc).__name__)
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 1:
            raise
        try:
            from app.database.connection import SessionLocal
            from app.services.gmail_mcp_server import search_mailbox
            max_messages = (settings.gmail_mcp_max_messages if full_scan else
                            min(settings.gmail_mcp_max_messages, MAX_LIVE_GMAIL_MESSAGES))
            with SessionLocal() as db:
                res = search_mailbox(
                    db, user_id, gmail_search_query(query, include_topic_terms=not full_scan),
                    max_messages, time_budget_seconds=remaining,
                )
                res["source"] = "gmail_direct"
                return res
        except Exception as direct_exc:
            logger.warning("Direct Gmail search fallback also failed: %s", type(direct_exc).__name__)
            raise exc


# Compatibility alias for callers that still use the old function name.
search_live_primary = search_live_mailbox

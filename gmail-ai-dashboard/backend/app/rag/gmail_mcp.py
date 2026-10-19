"""Account-bound MCP client for live Gmail discovery before local retrieval."""
import asyncio
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

from app.core.config import settings
from app.rag.search_plan import gmail_search_query


async def _search(user_id, query):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "app.services.gmail_mcp_server"],
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, "GMAIL_MCP_USER_ID": str(user_id)},
    )
    async with asyncio.timeout(settings.gmail_mcp_timeout_seconds):
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=settings.gmail_mcp_timeout_seconds)) as session:
                await session.initialize()
                result = await session.call_tool("search_primary_emails", {
                    "query": gmail_search_query(query),
                    "max_messages": settings.gmail_mcp_max_messages,
                })
                if result.isError:
                    raise RuntimeError("Live Gmail MCP search failed")
                payload = result.structuredContent
                if payload is None:
                    payload = json.loads(next(c.text for c in result.content if c.type == "text"))
                if not isinstance(payload.get("email_ids"), list) or type(payload.get("complete")) is not bool:
                    raise ValueError("Invalid Gmail MCP response")
                return payload


def search_live_primary(user_id, query):
    return asyncio.run(_search(user_id, query))

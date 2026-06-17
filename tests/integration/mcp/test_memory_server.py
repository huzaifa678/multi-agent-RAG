"""Integration tests for the memory MCP server."""

import pytest
from fastmcp import Client

import mcp_servers.memory.server as memory_server

pytestmark = pytest.mark.integration


async def test_registers_tools():
    async with Client(memory_server.mcp) as client:
        names = {t.name for t in await client.list_tools()}
    assert {"save_message", "get_history"} <= names


def test_save_message_persists(monkeypatch):
    calls = []
    monkeypatch.setattr(
        memory_server, "insert_message", lambda *a: calls.append(a)
    )

    result = memory_server.save_message("sess", "user", "hello", "gpt-4o-mini")

    assert result == {"status": "saved"}
    assert calls == [("sess", "user", "hello", "gpt-4o-mini")]


def test_get_history_wraps_rows(monkeypatch):
    rows = [{"role": "user", "content": "hi"}]
    monkeypatch.setattr(memory_server, "get_chat_history", lambda sid, limit: rows)

    result = memory_server.get_history("sess", limit=5)

    assert result == {"structured_content": rows}


async def test_get_history_through_client(monkeypatch):
    rows = [{"role": "assistant", "content": "yo"}]
    monkeypatch.setattr(memory_server, "get_chat_history", lambda sid, limit: rows)

    async with Client(memory_server.mcp) as client:
        res = await client.call_tool(
            "get_history", {"session_id": "sess", "limit": 3}
        )

    # Dict returns surface as structured data over the MCP transport.
    assert res.data == {"structured_content": rows}

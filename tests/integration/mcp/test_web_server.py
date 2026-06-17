"""Integration tests for the web MCP server."""

import pytest
from fastmcp import Client

import mcp_servers.web.server as web_server

pytestmark = pytest.mark.integration


async def test_registers_tools():
    async with Client(web_server.mcp) as client:
        names = {t.name for t in await client.list_tools()}
    assert "search_web" in names


def test_search_web_delegates(monkeypatch):
    monkeypatch.setattr(
        web_server, "web_search", lambda q: [{"source": "web", "content": q}]
    )

    assert web_server.search_web("rust") == [{"source": "web", "content": "rust"}]

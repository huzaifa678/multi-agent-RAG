"""Integration tests for the RAG MCP server."""

import pytest
from fastmcp import Client

import mcp_servers.rag.server as rag_server

pytestmark = pytest.mark.integration


async def test_registers_tools():
    async with Client(rag_server.mcp) as client:
        names = {t.name for t in await client.list_tools()}
    assert {"add_documents_tool", "retrieve_documents_tool"} <= names


def test_add_documents_delegates(monkeypatch):
    seen = {}

    def fake_add(docs):
        seen["docs"] = docs
        return {"added": len(docs)}

    monkeypatch.setattr(rag_server, "add_documents", fake_add)

    result = rag_server.add_documents_tool(["a", "b"])

    assert result == {"added": 2}
    assert seen["docs"] == ["a", "b"]


def test_retrieve_forwards_top_k(monkeypatch):
    captured = {}

    def fake_retrieve(query, top_k=5):
        captured["query"] = query
        captured["top_k"] = top_k
        return [{"content": "doc"}]

    monkeypatch.setattr(rag_server, "retrieve_documents", fake_retrieve)

    result = rag_server.retrieve_documents_tool("what is rust?", top_k=3)

    assert result == [{"content": "doc"}]
    assert captured == {"query": "what is rust?", "top_k": 3}

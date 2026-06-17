"""Integration tests for the RAG agent's run flow (store + LLM mocked)."""

import asyncio

import pytest

import agents.rag_agent as rag_agent
from tests.conftest import FakeChain

pytestmark = pytest.mark.integration


async def test_run_rag_found_path(monkeypatch):
    monkeypatch.setattr(
        rag_agent, "retrieve_context", lambda q: [{"content": "stored doc"}]
    )
    chain = FakeChain("answer from context")
    monkeypatch.setattr(rag_agent, "rag_chain", chain)

    result = await rag_agent.run_rag("question")

    assert result == {
        "content": "answer from context",
        "source": "rag",
        "found": True,
    }
    assert chain.calls[0]["context"] == "stored doc"


async def test_run_rag_not_found_without_web_context(monkeypatch):
    monkeypatch.setattr(rag_agent, "retrieve_context", lambda q: [])
    chain = FakeChain("not found in knowledge base")
    monkeypatch.setattr(rag_agent, "rag_chain", chain)

    result = await rag_agent.run_rag("question")

    assert result["found"] is False
    # When nothing is retrieved, the sentinel context is sent to the LLM.
    assert chain.calls[0]["context"] == "NOT_FOUND"


async def test_run_rag_learns_from_web_context(monkeypatch):
    # First retrieval misses, then succeeds after ingesting web context.
    retrievals = iter([[], [{"content": "newly learned"}]])
    monkeypatch.setattr(rag_agent, "retrieve_context", lambda q: next(retrievals))
    monkeypatch.setattr(rag_agent, "clean_text", lambda t: t)
    monkeypatch.setattr(rag_agent, "chunk_text", lambda t: ["chunk-1"])

    added = {}

    class FakeStore:
        def add_texts(self, texts, metadatas):
            added["texts"] = texts
            added["metadatas"] = metadatas

    monkeypatch.setattr(rag_agent, "vectorstore", FakeStore())
    monkeypatch.setattr(rag_agent, "rag_chain", FakeChain("answer"))

    result = await rag_agent.run_rag("q", web_context="useful web info")

    # Let the fire-and-forget ingestion task run.
    await asyncio.sleep(0.05)

    assert result["found"] is True
    assert result["content"] == "answer"
    assert added["texts"] == ["chunk-1"]
    assert added["metadatas"][0]["source"] == "auto_update"


async def test_run_rag_skips_learning_when_web_has_no_relevant(monkeypatch):
    monkeypatch.setattr(rag_agent, "retrieve_context", lambda q: [])
    monkeypatch.setattr(rag_agent, "rag_chain", FakeChain("x"))

    called = {"add": False}

    class Store:
        def add_texts(self, *a, **k):
            called["add"] = True

    monkeypatch.setattr(rag_agent, "vectorstore", Store())

    result = await rag_agent.run_rag("q", web_context="No relevant data here")

    assert result["found"] is False
    assert called["add"] is False

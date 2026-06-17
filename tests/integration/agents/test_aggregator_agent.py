"""Integration tests for the aggregator agent (DB + LLM I/O mocked)."""

import asyncio

import pytest

import agents.aggregator_agent as agg

pytestmark = pytest.mark.integration


def test_trim_empty_returns_empty():
    assert agg.trim("") == ""
    assert agg.trim(None) == ""


def test_trim_truncates_to_limit():
    assert agg.trim("abcdef", limit=3) == "abc"


def test_trim_default_limit():
    assert len(agg.trim("x" * 5000)) == agg.MAX_CONTEXT_CHARS


def test_build_long_term_memory_empty(monkeypatch):
    monkeypatch.setattr(agg, "get_long_term_memory", lambda *a, **k: [])
    assert agg.build_long_term_memory("s") == "No long-term memory found."


def test_build_long_term_memory_formats_rows(monkeypatch):
    rows = [
        {"source": "web", "content": "fact one"},
        {"source": None, "content": "fact two"},
    ]
    monkeypatch.setattr(agg, "get_long_term_memory", lambda *a, **k: rows)
    out = agg.build_long_term_memory("s")
    assert out == "web: fact one\nunknown: fact two"


def test_build_short_term_memory_uses_history(monkeypatch):
    history = [{"role": "user", "content": "hi"}]
    monkeypatch.setattr(agg, "get_chat_history", lambda *a, **k: history)
    assert agg.build_short_term_memory("s") == "user: hi"


def test_build_short_term_memory_no_memory_at_all(monkeypatch):
    monkeypatch.setattr(agg, "get_chat_history", lambda *a, **k: [])
    monkeypatch.setattr(agg, "get_long_term_memory", lambda *a, **k: [])
    assert agg.build_short_term_memory("s") == "No memory found."


def test_build_short_term_memory_bootstraps_from_long_term(monkeypatch):
    calls = iter([[], [{"role": "system", "content": "[BOOTSTRAP_MEMORY] x"}]])
    monkeypatch.setattr(agg, "get_chat_history", lambda *a, **k: next(calls))
    monkeypatch.setattr(
        agg, "get_long_term_memory", lambda *a, **k: [{"content": "x"}]
    )
    inserted = []
    monkeypatch.setattr(
        agg, "insert_message", lambda *a, **k: inserted.append((a, k))
    )

    out = agg.build_short_term_memory("s")

    assert "[BOOTSTRAP_MEMORY] x" in out
    # The long-term entry is bootstrapped into chat history exactly once.
    assert len(inserted) == 1


async def test_aggregate_response_returns_answer_and_persists(monkeypatch):
    async def fake_safe_call(payload):
        fake_safe_call.payload = payload
        return "FINAL ANSWER"

    monkeypatch.setattr(agg, "safe_llm_call", fake_safe_call)

    long_writes, msg_writes = [], []
    monkeypatch.setattr(
        agg, "insert_long_term_memory", lambda *a: long_writes.append(a)
    )
    monkeypatch.setattr(agg, "insert_message", lambda *a: msg_writes.append(a))

    state = {"rag": "R" * 2000, "web": "W", "memory": "M", "short_memory": "S"}
    answer = await agg.aggregate_response("q", "sess", state)

    assert answer == "FINAL ANSWER"
    # Oversized context is trimmed before reaching the LLM.
    assert len(fake_safe_call.payload["rag"]) == agg.MAX_CONTEXT_CHARS
    assert fake_safe_call.payload["query"] == "q"

    # Let the fire-and-forget persistence tasks run to completion.
    await asyncio.sleep(0.05)
    assert len(long_writes) == 1
    assert len(msg_writes) == 1
    assert msg_writes[0][1] == "assistant"


async def test_safe_llm_call_falls_back_on_error(monkeypatch):
    class Boom:
        async def ainvoke(self, _payload):
            raise RuntimeError("anthropic down")

    class Ok:
        async def ainvoke(self, _payload):
            return "fallback answer"

    monkeypatch.setattr(agg, "final_chain", Boom())
    monkeypatch.setattr(agg, "fallback_chain", Ok())

    out = await agg.safe_llm_call({"query": "q"})
    assert out == "fallback answer"

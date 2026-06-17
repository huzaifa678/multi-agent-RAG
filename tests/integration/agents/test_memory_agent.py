"""Integration tests for the memory agent's run flow (external I/O mocked)."""

import pytest

import agents.memory_agent as memory_agent
from tests.conftest import FakeChain

pytestmark = pytest.mark.integration


class FakeHistory:
    """Mimics the MCP tool response shape consumed by run_memory."""

    def __init__(self, structured_content):
        self.data = {"structured_content": structured_content}


def _patch_save(monkeypatch):
    saved = []

    async def fake_save(session_id, **kwargs):
        saved.append({"session_id": session_id, **kwargs})

    monkeypatch.setattr(memory_agent, "save_message", fake_save)
    return saved


async def test_run_memory_no_history_returns_no_memory(monkeypatch):
    async def empty_history(_session_id):
        return FakeHistory([])

    monkeypatch.setattr(memory_agent, "get_history", empty_history)
    saved = _patch_save(monkeypatch)

    result = await memory_agent.run_memory("sess-1")

    assert result == {"content": "NO_MEMORY", "source": "memory", "raw": ""}
    # Nothing meaningful to summarize, so nothing should be persisted.
    assert saved == []


async def test_run_memory_summarizes_and_saves(monkeypatch):
    history = [
        {"role": "user", "content": "I love Rust"},
        {"role": "assistant", "content": "Great choice"},
        {"role": "user", "content": ""},  # filtered out: empty content
    ]

    async def fake_history(_session_id):
        return FakeHistory(history)

    monkeypatch.setattr(memory_agent, "get_history", fake_history)
    monkeypatch.setattr(memory_agent, "memory_chain", FakeChain("User likes Rust."))
    saved = _patch_save(monkeypatch)

    result = await memory_agent.run_memory("sess-2")

    assert result["content"] == "User likes Rust."
    assert result["source"] == "memory"
    assert "I love Rust" in result["raw"]
    # Empty-content turn must be excluded from the formatted history.
    assert result["raw"].count("\n") == 1
    # The summary is persisted exactly once, tagged as a memory summary.
    assert len(saved) == 1
    assert saved[0]["content"] == "[MEMORY_SUMMARY] User likes Rust."


async def test_run_memory_handles_missing_data_attr(monkeypatch):
    async def raw_history(_session_id):
        return object()  # no .data attribute

    monkeypatch.setattr(memory_agent, "get_history", raw_history)
    _patch_save(monkeypatch)

    result = await memory_agent.run_memory("sess-3")

    assert result["content"] == "NO_MEMORY"

"""Integration tests for the web agent's run flow (external I/O mocked)."""

import pytest

import agents.web_agent as web_agent
from tests.conftest import FakeChain

pytestmark = pytest.mark.integration


# --- normalize() pure-function cases -------------------------------------

def test_normalize_none_returns_empty_string():
    assert web_agent.normalize(None) == ""


def test_normalize_plain_string():
    assert web_agent.normalize("hello") == "hello"


def test_normalize_object_with_content_attr():
    class Obj:
        content = "from-attr"

    assert web_agent.normalize(Obj()) == "from-attr"


def test_normalize_dict_prefers_content_then_results():
    assert web_agent.normalize({"content": "c", "results": "r"}) == "c"
    assert web_agent.normalize({"results": "r"}) == "r"


def test_normalize_list_of_dicts_and_scalars():
    out = web_agent.normalize([{"content": "a"}, "b", 3])
    assert out == "a\nb\n3"


# --- run_web() flow ------------------------------------------------------

async def _fake_search(_query):
    return [{"content": "Paris is the capital of France."}]


async def test_run_web_found_path(monkeypatch):
    monkeypatch.setattr(web_agent, "search_web", _fake_search)
    chain = FakeChain("Paris is the capital.")
    monkeypatch.setattr(web_agent, "web_chain", chain)

    result = await web_agent.run_web("capital of France?")

    assert result["source"] == "web"
    assert result["content"] == "Paris is the capital."
    assert result["found"] is True
    assert "Paris" in result["raw"]
    # The normalized search text must be forwarded to the summarization chain.
    assert chain.calls[0]["results"] == "Paris is the capital of France."


async def test_run_web_not_found_when_no_relevant(monkeypatch):
    async def empty_search(_query):
        return "No relevant results were found."

    monkeypatch.setattr(web_agent, "search_web", empty_search)
    monkeypatch.setattr(web_agent, "web_chain", FakeChain("nothing"))

    result = await web_agent.run_web("obscure query")

    assert result["found"] is False

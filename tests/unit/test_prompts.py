"""Unit tests for the centralized prompt builders in ``agents/prompts.py``."""

import pytest
from langchain_core.prompts import ChatPromptTemplate

from agents.prompts import (
    aggregator_final_prompt,
    aggregator_plan_prompt,
    aggregator_replan_prompt,
    memory_prompt,
    rag_prompt,
    web_prompt,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "builder",
    [
        aggregator_plan_prompt,
        aggregator_replan_prompt,
        aggregator_final_prompt,
        rag_prompt,
        web_prompt,
        memory_prompt,
    ],
)
def test_builder_returns_chat_prompt_template(builder):
    assert isinstance(builder(), ChatPromptTemplate)


def test_builders_return_fresh_instances():
    # Each call should produce an independent template, not a shared singleton.
    assert rag_prompt() is not rag_prompt()


def test_plan_prompt_input_variables():
    prompt = aggregator_plan_prompt()
    assert set(prompt.input_variables) == {"query", "short_memory", "long_memory"}


def test_replan_prompt_input_variables():
    prompt = aggregator_replan_prompt()
    assert set(prompt.input_variables) == {"query", "rag", "web", "memory"}


def test_final_prompt_input_variables():
    prompt = aggregator_final_prompt()
    assert set(prompt.input_variables) == {
        "query",
        "short_memory",
        "long_memory",
        "memory",
        "rag",
        "web",
    }


def test_rag_prompt_input_variables():
    assert set(rag_prompt().input_variables) == {"query", "context"}


def test_web_prompt_input_variables():
    assert set(web_prompt().input_variables) == {"query", "results"}


def test_memory_prompt_input_variables():
    assert set(memory_prompt().input_variables) == {"history"}


def test_plan_prompt_formats_with_values():
    # The literal "{{ }}" JSON braces in the system message must survive
    # formatting without being treated as template variables.
    messages = aggregator_plan_prompt().format_messages(
        query="hi", short_memory="s", long_memory="l"
    )
    rendered = "\n".join(m.content for m in messages)
    assert '"agent_calls"' in rendered
    assert "hi" in rendered


def test_rag_prompt_carries_grounding_instruction():
    system_text = rag_prompt().format_messages(query="q", context="c")[0].content
    assert "not found in knowledge base" in system_text

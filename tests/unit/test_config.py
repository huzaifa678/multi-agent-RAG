"""Unit tests for the ``AgentConfig`` LLM factory in ``agents/config.py``."""

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from agents.config import AgentConfig

pytestmark = pytest.mark.unit


def test_model_name_constants():
    assert AgentConfig.ANTHROPIC_MODEL == "claude-sonnet-4-6"
    assert AgentConfig.OPENAI_MODEL == "gpt-4o-mini"
    assert AgentConfig.GROQ_MODEL == "llama-3.3-70b-versatile"


def test_aggregator_llm_type_and_model():
    llm = AgentConfig.aggregator_llm()
    assert isinstance(llm, ChatAnthropic)
    assert llm.model == AgentConfig.ANTHROPIC_MODEL
    assert llm.temperature == 0
    assert llm.max_tokens == 1024


def test_aggregator_fallback_llm_type_and_model():
    llm = AgentConfig.aggregator_fallback_llm()
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == AgentConfig.OPENAI_MODEL
    assert llm.max_tokens == 400


def test_rag_llm_type_and_model():
    llm = AgentConfig.rag_llm()
    assert isinstance(llm, ChatGroq)
    assert llm.model_name == AgentConfig.GROQ_MODEL
    # Groq clamps an exact 0 to a tiny epsilon, so compare approximately.
    assert llm.temperature == pytest.approx(0, abs=1e-6)


def test_web_llm_type_and_model():
    llm = AgentConfig.web_llm()
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == AgentConfig.OPENAI_MODEL


def test_memory_llm_type_and_model():
    llm = AgentConfig.memory_llm()
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == AgentConfig.OPENAI_MODEL


def test_factories_return_new_instances():
    assert AgentConfig.web_llm() is not AgentConfig.web_llm()

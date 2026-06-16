"""
Centralized model initialization for all agents.

`AgentConfig` is the single place where LLM clients are constructed. Each agent
falls back to one of these factory methods when no LLM is injected, keeping
model names, temperatures and token limits out of the agent logic.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from utils.config import Config


class AgentConfig:
    """Factory for the LLM clients used across the agent system."""

    ANTHROPIC_MODEL = "claude-sonnet-4-6"
    OPENAI_MODEL = "gpt-4o-mini"
    GROQ_MODEL = "llama-3.3-70b-versatile"

    @staticmethod
    def aggregator_llm() -> ChatAnthropic:
        return ChatAnthropic(
            api_key=Config.ANTHROPIC_API_KEY,
            model=AgentConfig.ANTHROPIC_MODEL,
            temperature=0,
            max_tokens=1024,
        )

    @staticmethod
    def aggregator_fallback_llm() -> ChatOpenAI:
        return ChatOpenAI(
            api_key=Config.OPENAI_API_KEY,
            model=AgentConfig.OPENAI_MODEL,
            temperature=0,
            max_tokens=400,
        )

    @staticmethod
    def rag_llm() -> ChatGroq:
        return ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=AgentConfig.GROQ_MODEL,
            temperature=0,
        )

    @staticmethod
    def web_llm() -> ChatOpenAI:
        return ChatOpenAI(
            api_key=Config.OPENAI_API_KEY,
            model=AgentConfig.OPENAI_MODEL,
            temperature=0,
        )

    @staticmethod
    def memory_llm() -> ChatOpenAI:
        return ChatOpenAI(
            api_key=Config.OPENAI_API_KEY,
            model=AgentConfig.OPENAI_MODEL,
            temperature=0,
        )

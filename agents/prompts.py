"""
Centralized prompt definitions for all agents.

Every agent imports its default ChatPromptTemplate from here so that prompt
engineering is decoupled from agent logic and model initialization.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def aggregator_plan_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a STRICT JSON generator.

You MUST output ONLY valid JSON matching this schema:

{{
  "agent_calls": ["rag", "web", "memory"],
  "confidence": {{
      "rag": float,
      "web": float,
      "memory": float
  }}
}}

Rules:
- NO markdown
- NO explanation
- ONLY valid JSON

CRITICAL RULES:
- "memory" MUST ALWAYS be included in agent_calls
- agent_calls can NEVER be empty
- Minimum output: ["memory"]

Confidence rules:
- 0.0 to 1.0 for each tool
- memory confidence must be >= 0.7 unless session is invalid

Tool selection:
- rag → factual/internal knowledge
- web → real-time or external info
- memory → ALWAYS for personalization
""",
            ),
            (
                "human",
                """
Query: {query}

Short-term memory:
{short_memory}

Long-term memory:
{long_memory}
""",
            ),
        ]
    )


def aggregator_replan_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a ReAct evaluator.

Decide:
1. Are we done?
2. Do we need more tools?

RULES:
- Only mark 'done': true when you have sufficient information to answer the query.

STRATEGY:
1. If RAG says "not found" but Web has the answer, you MUST call "rag" again.
   This triggers the RAG agent to save the Web info into the database.

Return structured output only.
""",
            ),
            (
                "human",
                """
Query: {query}

RAG:
{rag}

Web:
{web}

Memory:
{memory}
""",
            ),
        ]
    )


def aggregator_final_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a final reasoning agent.

Use ONLY provided sources:
- RAG (highest priority)
- Web
- Memory

Rules:
- Never hallucinate missing facts
- If info is missing, say so
""",
            ),
            (
                "human",
                """
Query: {query}

Short-term memory:
{short_memory}

Long-term memory:
{long_memory}

Memory Tool Output:
{memory}

RAG:
{rag}

Web:
{web}
""",
            ),
        ]
    )


def rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer ONLY using context. If missing, say 'not found in knowledge base'.",
            ),
            ("human", "Query: {query}\n\nContext:\n{context}"),
        ]
    )


def web_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Use ONLY provided results Do NOT repeat generic definitions Avoid template-like explanations Focus on unique facts only",
            ),
            ("human", "Query: {query}\n\nResults:\n{results}"),
        ]
    )


def memory_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Analyze the provided conversation history.
1. If you find user preferences, goals, or personal context, list them as 'Stable Facts'.
2. If the conversation is a general topic (like Rust programming), provide a 1-sentence summary of the discussion topic.
3. Combine these into a concise memory entry.
4. If the history is completely empty or contains no meaningful information, return: NO_MEMORY
""",
            ),
            ("human", "{history}"),
        ]
    )

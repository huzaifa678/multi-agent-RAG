import asyncio

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agents.base import BaseAggregatorAgent
from memory.sqllite_memory import (
    get_chat_history,
    get_long_term_memory,
    insert_long_term_memory,
    insert_message,
)
from schemas.plan import PlanSchema
from schemas.replan import ReplanSchema
from utils.config import Config
from utils.logger import get_logger

logger = get_logger()

MAX_CONTEXT_CHARS = 1200


class AggregatorAgent(BaseAggregatorAgent):
    """
    Aggregator Agent: Plans tool usage, replans dynamically,
    and produces the final aggregated answer.

    Dependencies are injected via the constructor for testability
    and loose coupling.
    """

    def __init__(
        self,
        llm: ChatAnthropic = None,
        fallback_llm: ChatOpenAI = None,
        plan_prompt: ChatPromptTemplate = None,
        replan_prompt: ChatPromptTemplate = None,
        final_prompt: ChatPromptTemplate = None,
    ):
        self.llm = llm or ChatAnthropic(
            api_key=Config.ANTHROPIC_API_KEY,
            model="claude-sonnet-4-6",
            temperature=0,
            max_tokens=1024,
        )

        self._fallback_llm = fallback_llm or ChatOpenAI(
            api_key=Config.OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=400,
        )

        self.plan_prompt = plan_prompt or ChatPromptTemplate.from_messages(
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

        self.replan_prompt = replan_prompt or ChatPromptTemplate.from_messages(
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

        self.final_prompt = final_prompt or ChatPromptTemplate.from_messages(
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

        self.plan_chain = self.plan_prompt | self.llm.with_structured_output(
            PlanSchema
        )
        self.replan_chain = self.replan_prompt | self.llm.with_structured_output(
            ReplanSchema
        )
        self.final_chain = self.final_prompt | self.llm | StrOutputParser()
        self.fallback_chain = (
            self.final_prompt | self._fallback_llm | StrOutputParser()
        )

    @staticmethod
    def _trim(text: str, limit: int = MAX_CONTEXT_CHARS):
        if not text:
            return ""
        return text[:limit]

    def build_short_term_memory(self, session_id: str, limit: int = 10):
        history = get_chat_history(session_id, limit)

        if history:
            return "\n".join([f"{h['role']}: {h['content']}" for h in history])

        long_memory = get_long_term_memory(session_id, limit=5)

        if not long_memory:
            return "No memory found."

        for mem in long_memory:
            insert_message(
                session_id,
                role="system",
                content=f"[BOOTSTRAP_MEMORY] {mem['content']}",
                model_used="bootstrap",
            )

        history = get_chat_history(session_id, limit)

        return "\n".join([f"{h['role']}: {h['content']}" for h in history])

    def build_long_term_memory(self, session_id: str, limit: int = 20):
        history = get_long_term_memory(session_id, limit)
        if not history:
            return "No long-term memory found."
        return "\n".join(
            [f"{h['source'] or 'unknown'}: {h['content']}" for h in history]
        )

    async def _safe_llm_call(self, payload: dict):
        try:
            logger.info("Using anthropic")
            return await self.final_chain.ainvoke(payload)
        except Exception as e:
            logger.warning("Anthropic failed switching to OpenAI:", str(e))
            return await self.fallback_chain.ainvoke(payload)

    def plan(self, query: str, short_memory: str, long_memory: str):
        result = self.plan_chain.invoke(
            {
                "query": query,
                "short_memory": short_memory,
                "long_memory": long_memory,
            }
        )

        if not result.agent_calls:
            result.agent_calls = ["memory"]

        dynamic_confidence = {
            call.tool: call.confidence for call in result.agent_calls
        }

        tool_names = [call.tool for call in result.agent_calls]

        return {
            "short_memory": short_memory,
            "long_memory": long_memory,
            "agent_calls": tool_names,
            "executed_calls": [],
            "confidence": dynamic_confidence,
            "planner_confidence_log": {
                "raw_confidence": dynamic_confidence,
                "ordered_tools": tool_names,
            },
        }

    def replan(
        self,
        query: str,
        existing_calls: list,
        rag: str = "",
        web: str = "",
        memory: str = "",
    ):
        result = self.replan_chain.invoke(
            {
                "query": query,
                "rag": rag,
                "web": web,
                "memory": memory,
            }
        )

        new_calls = result.agent_calls or []
        updated_calls = list(set(existing_calls + new_calls))

        is_done = getattr(result, "done", False)

        return {
            "agent_calls": updated_calls,
            "done": is_done,
            "replan_debug": {"next_calls": new_calls, "done": is_done},
        }

    async def aggregate(self, query: str, session_id: str, state_data: dict):
        rag_content = self._trim(state_data.get("rag", ""))
        web_content = self._trim(state_data.get("web", ""))
        memory_content = self._trim(state_data.get("memory", ""))
        short_memory = self._trim(state_data.get("short_memory", ""), 800)
        long_memory = self._trim(state_data.get("long_memory", ""), 800)

        final_answer = await self._safe_llm_call(
            {
                "query": query,
                "short_memory": short_memory,
                "long_memory": long_memory,
                "memory": memory_content,
                "rag": rag_content,
                "web": web_content,
            }
        )

        asyncio.create_task(
            asyncio.to_thread(
                insert_long_term_memory,
                session_id,
                f"Q: {query} | A: {final_answer[:500]}",
                "hybrid",
            )
        )

        asyncio.create_task(
            asyncio.to_thread(
                insert_message,
                session_id,
                "assistant",
                final_answer,
                "claude-sonnet-4-6",
            )
        )

        return final_answer


aggregator_agent = AggregatorAgent()
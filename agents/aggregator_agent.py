from memory.sqllite_memory import (
    get_chat_history,
    get_long_term_memory,
    insert_long_term_memory,
    insert_message,
)

from schemas.plan import PlanSchema
from schemas.replan import ReplanSchema
import asyncio
from langchain_core.output_parsers import StrOutputParser
from utils.logger import get_logger
from agents.config import AgentConfig
from agents.prompts import (
    aggregator_plan_prompt,
    aggregator_replan_prompt,
    aggregator_final_prompt,
)

logger = get_logger()

llm = AgentConfig.aggregator_llm()

MAX_CONTEXT_CHARS = 1200

PLAN_PROMPT = aggregator_plan_prompt()

plan_chain = PLAN_PROMPT | llm.with_structured_output(PlanSchema)

REPLAN_PROMPT = aggregator_replan_prompt()

replan_chain = REPLAN_PROMPT | llm.with_structured_output(ReplanSchema)


FINAL_PROMPT = aggregator_final_prompt()

final_chain = FINAL_PROMPT | llm | StrOutputParser()


def build_short_term_memory(session_id: str, limit: int = 10):
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


def build_long_term_memory(session_id: str, limit: int = 20):
    history = get_long_term_memory(session_id, limit)
    if not history:
        return "No long-term memory found."
    return "\n".join([f"{h['source'] or 'unknown'}: {h['content']}" for h in history])


fallback_llm = AgentConfig.aggregator_fallback_llm()

fallback_chain = FINAL_PROMPT | fallback_llm | StrOutputParser()


async def safe_llm_call(payload):
    try:
        logger.info("Using anthropic")
        return await final_chain.ainvoke(payload)
    except Exception as e:
        logger.warning("Anthropic failed switching to OpenAI:", str(e))
        return await fallback_chain.ainvoke(payload)


def trim(text: str, limit: int = MAX_CONTEXT_CHARS):
    if not text:
        return ""
    return text[:limit]


async def aggregate_response(query: str, session_id: str, state_data: dict):

    rag_content = trim(state_data.get("rag", ""))
    web_content = trim(state_data.get("web", ""))
    memory_content = trim(state_data.get("memory", ""))
    short_memory = trim(state_data.get("short_memory", ""), 800)
    long_memory = trim(state_data.get("long_memory", ""), 800)

    final_answer = await safe_llm_call(
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
            insert_message, session_id, "assistant", final_answer, "claude-sonnet-4-6"
        )
    )

    return final_answer

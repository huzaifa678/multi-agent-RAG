from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.rag_agent import run_rag
from agents.web_agent import run_web
from agents.memory_agent import run_memory

from memory.sqllite_memory import (
    get_chat_history,
    get_long_term_memory,
    insert_long_term_memory
)

from schemas.plan import PlanSchema
from schemas.replan import ReplanSchema
from utils.config import Config


llm = ChatAnthropic(
    api_key=Config.ANTHROPIC_API_KEY,
    model="claude-sonnet-4-6",
    temperature=0,
    max_tokens=1024,
)


PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """
You are a STRICT ReAct planner.

You MUST select tools for any non-trivial query.

TOOL RULES:
- If query asks: what is, explain, how, why, define, tell me about → MUST use web or rag
- Only allow empty tool usage for greetings, small talk, or simple math

TOOLS:
- rag: internal knowledge base
- web: external real-time knowledge
- memory: user/session context

Return structured output only.
"""),
    ("human",
     """
Query: {query}

Short-term memory:
{short_memory}

Long-term memory:
{long_memory}
""")
])

plan_chain = PLAN_PROMPT | llm.with_structured_output(PlanSchema)

REPLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """
You are a ReAct evaluator.

Decide:
1. Are we done?
2. Do we need more tools?

Return structured output only.
"""),
    ("human",
     """
Query: {query}

RAG:
{rag}

Web:
{web}

Memory:
{memory}
""")
])

replan_chain = REPLAN_PROMPT | llm.with_structured_output(ReplanSchema)


FINAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """
You are a final reasoning agent.

Use ONLY provided sources:
- RAG (highest priority)
- Web
- Memory

Rules:
- Never hallucinate missing facts
- If info is missing, say so
"""),
    ("human",
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
""")
])

final_chain = FINAL_PROMPT | llm | StrOutputParser()


def build_short_term_memory(session_id: str, limit: int = 10):
    history = get_chat_history(session_id, limit)
    if not history:
        return "No short-term memory found."
    return "\n".join([f"{h['role']}: {h['content']}" for h in history])


def build_long_term_memory(session_id: str, limit: int = 20):
    history = get_long_term_memory(session_id, limit)
    if not history:
        return "No long-term memory found."
    return "\n".join([
        f"{h['source'] or 'unknown'}: {h['content']}"
        for h in history
    ])


def execute_tools(query, session_id, agent_calls, prev_results=None):
    results = prev_results or {
        "rag": "",
        "web": "",
        "memory": ""
    }

    if "rag" in agent_calls and not results["rag"]:
        results["rag"] = run_rag(query)["content"]

    if "web" in agent_calls and not results["web"]:
        results["web"] = run_web(query)["content"]

    if "memory" in agent_calls and not results["memory"]:
        results["memory"] = run_memory(session_id)["content"]

    return results


def should_force_tools(query: str, results: dict) -> bool:
    keywords = ["what is", "explain", "how", "why", "define", "tell me"]
    if any(k in query.lower() for k in keywords):
        if not results["rag"] and not results["web"]:
            return True
    return False


def aggregate_response(query: str, session_id: str, max_steps: int = 2):

    short_memory = build_short_term_memory(session_id)
    long_memory = build_long_term_memory(session_id)

    plan_result = plan_chain.invoke({
        "query": query,
        "short_memory": short_memory,
        "long_memory": long_memory
    })

    results = execute_tools(query, session_id, plan_result.agent_calls)

    for _ in range(max_steps - 1):

        replan_result = replan_chain.invoke({
            "query": query,
            "rag": results["rag"],
            "web": results["web"],
            "memory": results["memory"]
        })

        results = execute_tools(
            query,
            session_id,
            replan_result.agent_calls,
            results
        )

        if getattr(replan_result, "done", False):
            break

    if should_force_tools(query, results):
        results["web"] = run_web(query)["content"]

    final_answer = final_chain.invoke({
        "query": query,
        "short_memory": short_memory,
        "long_memory": long_memory,
        "memory": results["memory"] or "No relevant memory found.",
        "rag": results["rag"] or "No relevant RAG content found.",
        "web": results["web"] or "No relevant Web content found."
    })

    if final_answer:
        insert_long_term_memory(
            session_id,
            f"[react] Q: {query} | A: {final_answer[:1000]}",
            source="aggregator"
        )

    return {
        "content": final_answer,
        "source": "aggregator",
        "raw": {
            "plan": plan_result.model_dump(),
            "results": results
        }
    }
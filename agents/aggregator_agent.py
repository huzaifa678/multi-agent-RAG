import json

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

from utils.config import Config

llm = ChatAnthropic(
    api_key=Config.ANTHROPIC_API_KEY,
    model="claude-3-5-sonnet-20241022",
    temperature=0
)


PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """
You are a ReAct planner.

Return ONLY valid JSON:

{
  "thought": "reasoning",
  "agent_calls": ["rag", "web", "memory"]
}

Rules:
- Use RAG for factual/internal data
- Use Web for latest/current info
- Use Memory for user/session context
- Choose only necessary tools
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


plan_chain = PLAN_PROMPT | llm | StrOutputParser()


REPLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """
You are a ReAct evaluator.

Given tool results, decide if more tools are needed.

Return ONLY JSON:

{
  "thought": "...",
  "agent_calls": []
}

Rules:
- If info is sufficient → return empty agent_calls
- If missing data → request additional tools
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


replan_chain = REPLAN_PROMPT | llm | StrOutputParser()


FINAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """
You are a final reasoning agent.

Combine all sources:
- RAG (highest priority)
- Web
- Memory

Rules:
- Resolve conflicts carefully
- If missing info, say so
- Do NOT expose reasoning
"""),
    ("human",
     """
Query: {query}

Short-term memory:
{short_memory}

Long-term memory:
{long_memory}

Memory Agent:
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


def parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return {"agent_calls": []}


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


def aggregate_response(query: str, session_id: str, max_steps: int = 2):
    short_memory = build_short_term_memory(session_id)
    long_memory = build_long_term_memory(session_id)

    plan_raw = plan_chain.invoke({
        "query": query,
        "short_memory": short_memory,
        "long_memory": long_memory
    })

    plan_data = parse_json(plan_raw)
    agent_calls = plan_data.get("agent_calls", [])

    results = execute_tools(query, session_id, agent_calls)

    for _ in range(max_steps - 1):

        replan_raw = replan_chain.invoke({
            "query": query,
            "rag": results["rag"],
            "web": results["web"],
            "memory": results["memory"]
        })

        replan_data = parse_json(replan_raw)
        new_calls = replan_data.get("agent_calls", [])

        if not new_calls:
            break

        results = execute_tools(query, session_id, new_calls, results)

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
            "plan": plan_data,
            "results": results
        }
    }
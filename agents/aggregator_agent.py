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
- NO text outside JSON
- ONLY valid JSON
- "confidence": A score from 0.0 to 1.0 for EVERY tool listed in agent_calls. 
- Higher confidence (0.8+) for tools directly matching the query type (e.g., 'memory' for personal history, 'web' for news).
- If unsure, return empty list for agent_calls and 0.0 for confidence.

IMPORTANT RULE:
- ALWAYS include "memory" in agent_calls (it is mandatory for every query)
- Set confidence["memory"] >= 0.7 unless session_id is empty or invalid

Reason:
Memory is required to personalize and improve response quality.

If unsure:
- still include "memory"
- just lower confidence (0.3–0.6)
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

RULES:
- Only mark 'done': true when you have sufficient information to answer the query.

STRATEGY:
1. If RAG says "not found" but Web has the answer, you MUST call "rag" again. 
   This triggers the RAG agent to save the Web info into the database.

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


def execute_tools(query, session_id, agent_calls):
    results = results or {"rag": None, "web": None, "memory": None}

    if "rag" in agent_calls and not results["rag"]:
        results["rag"] = run_rag(query)

    if "web" in agent_calls and not results["web"]:
        results["web"] = run_web(query)

    if "memory" in agent_calls and not results["memory"]:
        results["memory"] = run_memory(session_id)

    return results

def rag_failed(rag_output: str) -> bool:
    return (
        not rag_output
        or "not found" in rag_output.lower()
        or "no relevant" in rag_output.lower()
    )


def web_has_answer(web_output: str) -> bool:
    return bool(web_output and "no relevant" not in web_output.lower())


def should_force_web(query: str, results: dict) -> bool:
    keywords = ["what", "how", "why", "explain", "define"]
    return any(k in query.lower() for k in keywords) and not results["web"]


async def aggregate_response(query: str, session_id: str, max_steps: int = 3):

    short_memory = get_chat_history(session_id)

    plan = ["rag", "web", "memory"]  

    results = execute_tools(query, session_id, plan)

    for _ in range(max_steps - 1):

        rag_state = results["rag"]
        web_state = results["web"]
        memory_state = results["memory"]

        if rag_failed(rag_state) and web_has_answer(web_state):

            results["rag"] = await run_rag(
                query,
                web_context=web_state["content"]
            )

        if (rag_state and rag_state.get("found")) or web_state:
            break

    final_answer = final_chain.invoke({
        "query": query,
        "memory": short_memory,
        "rag": results["rag"]["content"] if results["rag"] else "",
        "web": results["web"]["content"] if results["web"] else "",
        "memory": results["memory"]["content"] if results["memory"] else ""
    })

    insert_long_term_memory(
        session_id,
        f"[hybrid] Q: {query} | A: {final_answer[:1000]}",
        source="hybrid"
    )

    return {
        "content": final_answer,
        "raw": results
    }
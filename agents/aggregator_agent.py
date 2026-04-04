from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.rag_agent import run_rag
from agents.web_agent import run_web
from agents.memory_agent import run_memory

from memory.sqllite_memory import (
    get_chat_history,
    get_long_term_memory,
    insert_long_term_memory,
    insert_message
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

    if history:
        return "\n".join([
            f"{h['role']}: {h['content']}" for h in history
        ])

    long_memory = get_long_term_memory(session_id, limit=5)

    if not long_memory:
        return "No memory found."

    for mem in long_memory:
        insert_message(
            session_id,
            role="system",
            content=f"[BOOTSTRAP_MEMORY] {mem['content']}",
            model_used="bootstrap"
        )

    history = get_chat_history(session_id, limit)

    return "\n".join([
        f"{h['role']}: {h['content']}" for h in history
    ])


def build_long_term_memory(session_id: str, limit: int = 20):
    history = get_long_term_memory(session_id, limit)
    if not history:
        return "No long-term memory found."
    return "\n".join([
        f"{h['source'] or 'unknown'}: {h['content']}"
        for h in history
    ])


async def execute_tools(query, session_id, agent_calls):
    results = {"rag": None, "web": None, "memory": None}

    if "rag" in agent_calls:
        results["rag"] = await run_rag(query)

    if "web" in agent_calls:
        results["web"] = await run_web(query)

    if "memory" in agent_calls:
        results["memory"] = await run_memory(session_id)

    return results


def rag_failed(rag_output) -> bool:
    if not rag_output:
        return True

    if isinstance(rag_output, dict):
        rag_output = (
            rag_output.get("content")
            or rag_output.get("text")
            or ""
        )

    if hasattr(rag_output, "__await__"):
        return True

    if not isinstance(rag_output, str):
        return True

    return (
        "not found" in rag_output.lower()
        or "no relevant" in rag_output.lower()
    )


def web_has_answer(web_output) -> bool:
    if not web_output:
        return False

    if isinstance(web_output, dict):
        web_output = web_output.get("content") or ""

    if not isinstance(web_output, str):
        return False

    return "no relevant" not in web_output.lower()


def should_force_web(query: str, results: dict) -> bool:
    keywords = ["what", "how", "why", "explain", "define"]
    return any(k in query.lower() for k in keywords) and not results["web"]


async def aggregate_response(query: str, session_id: str, max_steps: int = 3):

    plan = ["rag", "web", "memory"]  

    results = await execute_tools(query, session_id, plan)

    for _ in range(max_steps - 1):

        rag_state = results["rag"]
        web_state = results["web"]

        if rag_failed(rag_state) and web_has_answer(web_state):

            results["rag"] = await run_rag(
                query,
                web_context=web_state["content"]
            )

        if (isinstance(rag_state, dict) and rag_state.get("found")) or (isinstance(web_state, dict) and web_state.get("content")):
            break

    final_answer = final_chain.invoke({
        "query": query,
        "short_memory": build_short_term_memory(session_id),
        "long_memory": build_long_term_memory(session_id),
        "memory": results["memory"]["content"] if results["memory"] else "",
        "rag": results["rag"]["content"] if results["rag"] else "",
        "web": results["web"]["content"] if results["web"] else ""
    })

    insert_long_term_memory(
        session_id,
        f"[hybrid] Q: {query} | A: {final_answer[:1000]}",
        source="hybrid"
    )

    insert_message(session_id, "user", query, model_used="user")

    insert_message(
        session_id,
        "assistant",
        final_answer,
        model_used="claude-sonnet-4-6"
    )

    return {
        "content": final_answer
    }
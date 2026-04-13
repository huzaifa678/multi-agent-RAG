import operator
from typing import Annotated, Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from agents.aggregator_agent import (
    aggregate_response,
    build_short_term_memory,
    build_long_term_memory,
    plan_chain,
    replan_chain,
)

from core import runtime
from agents.rag_agent import run_rag
from agents.web_agent import run_web
from agents.memory_agent import run_memory


class WorkflowState(TypedDict, total=False):
    query: str
    session_id: str

    short_memory: str
    long_memory: str

    agent_calls: Annotated[List[str], operator.add]
    executed_calls: Annotated[List[str], operator.add]
    confidence: Dict[str, float]

    rag: str
    web: str
    memory: str

    done: bool

    final_response: str


def planner_node(state: WorkflowState):
    short_memory = build_short_term_memory(state["session_id"])
    long_memory = build_long_term_memory(state["session_id"])

    plan_result = plan_chain.invoke(
        {
            "query": state["query"],
            "short_memory": short_memory,
            "long_memory": long_memory,
        }
    )

    if not plan_result.agent_calls:
        plan_result.agent_calls = ["memory"]

    dynamic_confidence = {
        call.tool: call.confidence for call in plan_result.agent_calls
    }

    tool_names = [call.tool for call in plan_result.agent_calls]

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


def route_tools(state: WorkflowState):
    if state.get("done"):
        return "aggregator"

    calls = state.get("agent_calls", [])
    executed = state.get("executed_calls", [])
    rag_content = state.get("rag", "").lower()

    if "web" in executed and "rag" not in executed:
        return "rag"

    if "web" in executed and "memory" not in executed and "memory" in calls:
        return "memory"

    if "memory" not in executed:
        return "memory"

    to_run = [c for c in calls if c not in executed]

    if not to_run:
        return "aggregator"

    return to_run


async def rag_node(state: WorkflowState):
    web_data = state.get("web")
    result = await run_rag(state["query"], web_data)

    executed = state.get("executed_calls", []) + ["rag"]

    return {"rag": result["content"], "executed_calls": executed}


async def web_node(state: WorkflowState):
    result = await run_web(state["query"])

    executed = state.get("executed_calls", []) + ["web"]

    return {"web": result["content"], "executed_calls": executed}


async def memory_node(state: WorkflowState):
    result = await run_memory(state["session_id"])

    executed = state.get("executed_calls", []) + ["memory"]

    return {"memory": result["content"], "executed_calls": executed}


def replan_node(state: WorkflowState):

    existing_calls = state.get("agent_calls", [])

    result = replan_chain.invoke(
        {
            "query": state["query"],
            "rag": state.get("rag", ""),
            "web": state.get("web", ""),
            "memory": state.get("memory", ""),
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


async def aggregator_node(state: WorkflowState):

    response_text = await aggregate_response(
        query=state["query"], session_id=state.get("session_id"), state_data=state
    )

    return {"final_response": response_text}


def build_workflow_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("planner", planner_node)
    graph.add_node("rag", rag_node)
    graph.add_node("web", web_node)
    graph.add_node("memory", memory_node)
    graph.add_node("replan", replan_node)
    graph.add_node("aggregator", aggregator_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_tools)

    graph.add_edge("rag", "replan")
    graph.add_edge("web", "replan")
    graph.add_edge("memory", "replan")

    graph.add_conditional_edges(
        "replan",
        route_tools,
        {
            "rag": "rag",
            "web": "web",
            "memory": "memory",
            "replan": "replan",
            "aggregator": "aggregator",
        },
    )

    graph.add_edge("aggregator", END)

    return graph.compile()


@traceable(name="langgraph_workflow")
async def execute_workflow(query: str, session_id: str) -> Dict[str, Any]:

    state = await runtime.app_graph.ainvoke({"query": query, "session_id": session_id})

    return {
        "response": state.get("final_response", ""),
        "debug": {
            "rag": state.get("rag"),
            "web": state.get("web"),
            "memory": state.get("memory"),
            "agent_calls": state.get("agent_calls"),
            "executed_calls": state.get("executed_calls"),
        },
    }

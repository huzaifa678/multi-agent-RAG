from typing import Dict, Any, List, TypedDict
import json
from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from agents.aggregator_agent import (
    build_short_term_memory,
    build_long_term_memory,
    plan_chain,
    replan_chain,
    final_chain
)

from agents.rag_agent import run_rag
from agents.web_agent import run_web
from agents.memory_agent import run_memory
from core import runtime
from memory.sqllite_memory import insert_long_term_memory


class WorkflowState(TypedDict, total=False):
    query: str
    session_id: str

    short_memory: str
    long_memory: str

    agent_calls: List[str]

    rag: str
    web: str
    memory: str

    final_response: str


def parse_json(text: str):
    try:
        return json.loads(text)
    except:
        return {"agent_calls": []}


def planner_node(state: WorkflowState):
    short_memory = build_short_term_memory(state["session_id"])
    long_memory = build_long_term_memory(state["session_id"])

    plan_raw = plan_chain.invoke({
        "query": state["query"],
        "short_memory": short_memory,
        "long_memory": long_memory
    })

    data = parse_json(plan_raw)

    return {
        "short_memory": short_memory,
        "long_memory": long_memory,
        "agent_calls": data.get("agent_calls", [])
    }


def route_tools(state: WorkflowState):
    calls = state.get("agent_calls", [])

    routes = []
    if "rag" in calls:
        routes.append("rag")
    if "web" in calls:
        routes.append("web")
    if "memory" in calls:
        routes.append("memory")

    return routes if routes else ["replan"]


def rag_node(state: WorkflowState):
    if not state.get("rag"):
        return {"rag": run_rag(state["query"])["content"]}
    return {}


def web_node(state: WorkflowState):
    if not state.get("web"):
        return {"web": run_web(state["query"])["content"]}
    return {}


def memory_node(state: WorkflowState):
    if not state.get("memory"):
        return {"memory": run_memory(state["session_id"])["content"]}
    return {}


def replan_node(state: WorkflowState):
    replan_raw = replan_chain.invoke({
        "query": state["query"],
        "rag": state.get("rag", ""),
        "web": state.get("web", ""),
        "memory": state.get("memory", "")
    })

    data = parse_json(replan_raw)

    return {
        "agent_calls": data.get("agent_calls", [])
    }


def aggregator_node(state: WorkflowState):
    final_answer = final_chain.invoke({
        "query": state["query"],
        "short_memory": state.get("short_memory", ""),
        "long_memory": state.get("long_memory", ""),
        "memory": state.get("memory", "No relevant memory found."),
        "rag": state.get("rag", "No relevant RAG content found."),
        "web": state.get("web", "No relevant Web content found.")
    })

    if state.get("session_id") and final_answer:
        insert_long_term_memory(
            state["session_id"],
            f"[hybrid-react] Q: {state['query']} | A: {final_answer[:1000]}",
            source="hybrid-react"
        )

    return {
        "final_response": final_answer
    }


def build_workflow_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("planner", planner_node)
    graph.add_node("rag", rag_node)
    graph.add_node("web", web_node)
    graph.add_node("memory", memory_node)
    graph.add_node("replan", replan_node)
    graph.add_node("aggregator", aggregator_node)

    graph.add_edge(START, "planner")

    graph.add_conditional_edges(
        "planner",
        route_tools,
        {
            "rag": "rag",
            "web": "web",
            "memory": "memory",
            "replan": "replan"
        }
    )

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
            "replan": "aggregator" 
        }
    )

    graph.add_edge("rag", "aggregator")
    graph.add_edge("web", "aggregator")
    graph.add_edge("memory", "aggregator")

    graph.add_edge("aggregator", END)

    return graph.compile()


@traceable(name="langgraph_workflow")
def execute_workflow(query: str, session_id: str) -> Dict[str, Any]:

    state = runtime.app_graph.invoke({
        "query": query,
        "session_id": session_id
    })

    return {
        "response": state.get("final_response", ""),
        "debug": {
            "rag": state.get("rag"),
            "web": state.get("web"),
            "memory": state.get("memory"),
            "agent_calls": state.get("agent_calls")
        }
    }
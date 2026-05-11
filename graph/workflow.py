import operator
from typing import Annotated, Dict, Any, List, TypedDict

from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from agents import WorkflowNodes
from agents.base import BaseWorkflow, BaseWorkflowNodes


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


class Workflow(BaseWorkflow):
    """
    OOP representation of the LangGraph workflow.

    Accepts WorkflowNodes via constructor dependency injection,
    avoiding implicit singleton instantiation at import time.
    This keeps the graph decoupled from the concrete agent layer.
    """

    def __init__(self, nodes: BaseWorkflowNodes = None):
        self.nodes = nodes or WorkflowNodes()

    @staticmethod
    def route_tools(state: WorkflowState):
        if state.get("done"):
            return "aggregator"

        calls = state.get("agent_calls", [])
        executed = state.get("executed_calls", [])
        state.get("rag", "").lower()

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

    async def run(self, query: str, session_id: str) -> Dict[str, Any]:
        graph = StateGraph(WorkflowState)

        graph.add_node("planner", self.nodes.planner_node)
        graph.add_node("rag", self.nodes.rag_node)
        graph.add_node("web", self.nodes.web_node)
        graph.add_node("memory", self.nodes.memory_node)
        graph.add_node("replan", self.nodes.replan_node)
        graph.add_node("aggregator", self.nodes.aggregator_node)

        graph.add_edge(START, "planner")
        graph.add_conditional_edges("planner", self.route_tools)

        graph.add_edge("rag", "replan")
        graph.add_edge("web", "replan")
        graph.add_edge("memory", "replan")

        graph.add_conditional_edges(
            "replan",
            self.route_tools,
            {
                "rag": "rag",
                "web": "web",
                "memory": "memory",
                "replan": "replan",
                "aggregator": "aggregator",
            },
        )

        graph.add_edge("aggregator", END)

        app = graph.compile()

        state = await app.ainvoke(
            {"query": query, "session_id": session_id}
        )

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


@traceable(name="langgraph_workflow")
async def execute_workflow(
    query: str,
    session_id: str,
    workflow: BaseWorkflow = None,
) -> Dict[str, Any]:
    """
    Backward-compatible function wrapper.
    Accepts an optional pre-built workflow instance, or creates one
    with default nodes for zero-config usage.
    """
    if workflow is None:
        workflow = Workflow()
    return await workflow.run(query, session_id)
"""
Shared workflow holder and DI registry.

The Workflow instance is created and registered at application startup
in main.py. This module acts as the single access point for the
shared workflow, avoiding circular imports.
"""

from typing import Optional

from agents.base import create_agent_config
from agents import WorkflowNodes
from graph.workflow import Workflow


_workflow: Optional[Workflow] = None


def set_workflow(workflow: Workflow):
    """Register the shared workflow instance."""
    global _workflow
    _workflow = workflow


def get_workflow() -> Workflow:
    """Get the shared workflow instance (lazily creates if unset)."""
    global _workflow
    if _workflow is None:
        _workflow = Workflow()
    return _workflow


async def get_workflow_async() -> Workflow:
    """Async-compatible dependency provider (for FastAPI Depends)."""
    return get_workflow()


def create_workflow_with_injected_agents(
    rag_agent=None,
    web_agent=None,
    memory_agent=None,
    aggregator_agent=None,
) -> Workflow:
    """
    Factory to create a Workflow with injected agent dependencies.
    Use this for testing or custom configurations.
    """
    config = create_agent_config(
        rag_agent=rag_agent,
        web_agent=web_agent,
        memory_agent=memory_agent,
        aggregator_agent=aggregator_agent,
    )
    nodes = WorkflowNodes(
        rag_agent=config["rag_agent"],
        web_agent=config["web_agent"],
        memory_agent=config["memory_agent"],
        aggregator_agent=config["aggregator_agent"],
    )
    return Workflow(nodes=nodes)
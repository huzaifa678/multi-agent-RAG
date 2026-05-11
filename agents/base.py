from abc import ABC, abstractmethod
from typing import Any


def create_agent_config(
    rag_agent=None,
    web_agent=None,
    memory_agent=None,
    aggregator_agent=None,
):
    """
    Factory function to create a configured WorkflowNodes + agents dict.
    All parameters are optional — defaults to importing the singleton
    instances from each agent module.
    """
    from agents.rag_agent import rag_agent as _rag
    from agents.web_agent import web_agent as _web
    from agents.memory_agent import memory_agent as _memory
    from agents.aggregator_agent import aggregator_agent as _aggregator

    return {
        "rag_agent": rag_agent or _rag,
        "web_agent": web_agent or _web,
        "memory_agent": memory_agent or _memory,
        "aggregator_agent": aggregator_agent or _aggregator,
    }


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Enforces a unified run interface across all agent implementations.
    """

    @abstractmethod
    async def run(self, **kwargs) -> Any:
        """Execute the agent's primary function."""
        ...


class BaseAggregatorAgent(ABC):
    """
    Abstract base class for the aggregator agent.
    Aggregators have multiple responsibilities (planning, replanning,
    aggregation) beyond a simple run() call.
    """

    @abstractmethod
    async def plan(self, query: str, short_memory: str, long_memory: str) -> dict:
        """Plan which tools to invoke for the given query."""
        ...

    @abstractmethod
    async def replan(
        self,
        query: str,
        existing_calls: list,
        rag: str = "",
        web: str = "",
        memory: str = "",
    ) -> dict:
        """Re-evaluate tool selection based on results so far."""
        ...

    @abstractmethod
    async def aggregate(
        self, query: str, session_id: str, state_data: dict
    ) -> str:
        """Produce the final aggregated answer."""
        ...

    @abstractmethod
    def build_short_term_memory(self, session_id: str, limit: int = 10) -> str:
        """Build short-term memory string from chat history."""
        ...

    @abstractmethod
    def build_long_term_memory(self, session_id: str, limit: int = 20) -> str:
        """Build long-term memory string."""
        ...


class BaseWorkflowNodes(ABC):
    """
    Abstract base class for workflow node definitions.
    Each node maps to a LangGraph node function signature.
    """

    @abstractmethod
    async def rag_node(self, state: dict) -> dict:
        ...

    @abstractmethod
    async def web_node(self, state: dict) -> dict:
        ...

    @abstractmethod
    async def memory_node(self, state: dict) -> dict:
        ...

    @abstractmethod
    def planner_node(self, state: dict) -> dict:
        ...

    @abstractmethod
    def replan_node(self, state: dict) -> dict:
        ...

    @abstractmethod
    async def aggregator_node(self, state: dict) -> dict:
        ...


class BaseWorkflow(ABC):
    """
    Abstract base class for the workflow orchestrator.
    """

    @abstractmethod
    async def run(self, query: str, session_id: str) -> dict:
        """Execute the full workflow for a given query and session."""
        ...
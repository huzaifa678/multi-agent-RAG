from agents.base import BaseWorkflowNodes
from agents.base import BaseAgent, BaseAggregatorAgent
from typing import Optional


class WorkflowNodes(BaseWorkflowNodes):
    """
    Concrete workflow node implementations.
    Each node delegates to the appropriate agent instance.

    Dependencies on agents are injected via the constructor,
    avoiding circular imports and enabling testability.
    """

    def __init__(
        self,
        rag_agent: Optional[BaseAgent] = None,
        web_agent: Optional[BaseAgent] = None,
        memory_agent: Optional[BaseAgent] = None,
        aggregator_agent: Optional[BaseAggregatorAgent] = None,
    ):
        self._rag = rag_agent
        self._web = web_agent
        self._memory = memory_agent
        self._aggregator = aggregator_agent

    async def rag_node(self, state: dict) -> dict:
        from agents.rag_agent import rag_agent

        web_data = state.get("web")
        agent = self._rag or rag_agent
        result = await agent.run(state["query"], web_data)

        executed = state.get("executed_calls", []) + ["rag"]

        return {"rag": result["content"], "executed_calls": executed}

    async def web_node(self, state: dict) -> dict:
        from agents.web_agent import web_agent

        agent = self._web or web_agent
        result = await agent.run(state["query"])

        executed = state.get("executed_calls", []) + ["web"]

        return {"web": result["content"], "executed_calls": executed}

    async def memory_node(self, state: dict) -> dict:
        from agents.memory_agent import memory_agent

        agent = self._memory or memory_agent
        result = await agent.run(state["session_id"])

        executed = state.get("executed_calls", []) + ["memory"]

        return {"memory": result["content"], "executed_calls": executed}

    def planner_node(self, state: dict) -> dict:
        from agents.aggregator_agent import aggregator_agent

        agent = self._aggregator or aggregator_agent

        short_memory = agent.build_short_term_memory(state["session_id"])
        long_memory = agent.build_long_term_memory(state["session_id"])

        result = agent.plan(
            query=state["query"],
            short_memory=short_memory,
            long_memory=long_memory,
        )

        return result

    def replan_node(self, state: dict) -> dict:
        from agents.aggregator_agent import aggregator_agent

        agent = self._aggregator or aggregator_agent
        existing_calls = state.get("agent_calls", [])

        result = agent.replan(
            query=state["query"],
            existing_calls=existing_calls,
            rag=state.get("rag", ""),
            web=state.get("web", ""),
            memory=state.get("memory", ""),
        )

        result["agent_calls"] = list(
            set(existing_calls + (result.get("agent_calls") or []))
        )
        return result

    async def aggregator_node(self, state: dict) -> dict:
        from agents.aggregator_agent import aggregator_agent

        agent = self._aggregator or aggregator_agent
        response_text = await agent.aggregate(
            query=state["query"],
            session_id=state.get("session_id"),
            state_data=state,
        )

        return {"final_response": response_text}
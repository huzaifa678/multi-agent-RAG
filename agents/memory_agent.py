from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agents.base import BaseAgent
from agents.config import AgentConfig
from agents.prompts import memory_prompt
from tools.memory.memory_tools import get_history, save_message
from utils.logger import get_logger

logger = get_logger()


class MemoryAgent(BaseAgent):
    """
    Memory Agent: Retrieves and summarizes conversation history
    to provide context-aware memory for the agent system.

    Dependencies are injected via the constructor for testability
    and loose coupling.
    """

    def __init__(
        self,
        llm: ChatOpenAI = None,
        prompt: ChatPromptTemplate = None,
        get_history_fn=None,
        save_message_fn=None,
    ):
        self.llm = llm or AgentConfig.memory_llm()

        self.prompt = prompt or memory_prompt()

        self.chain = self.prompt | self.llm | StrOutputParser()
        self._get_history_fn = get_history_fn or get_history
        self._save_message_fn = save_message_fn or save_message

    async def run(self, session_id: str):
        raw = await self._get_history_fn(session_id)

        history_list = []

        if hasattr(raw, "data") and isinstance(raw.data, dict):
            logger.info("The attribute is data")
            history_list = raw.data.get("structured_content") or []

        if not history_list or not isinstance(history_list, list):
            logger.warning("No content to use the memory is empty")
            return {"content": "NO_MEMORY", "source": "memory", "raw": ""}

        valid_items = [h for h in history_list if isinstance(h, dict) and h.get("content")]

        formatted = "\n".join(
            f"{h.get('role', 'unknown')}: {h.get('content', '')}" for h in valid_items
        )

        if not formatted.strip():
            return {"content": "NO_MEMORY", "source": "memory", "raw": ""}

        summary = await self.chain.ainvoke({"history": formatted})

        await self._save_message_fn(
            session_id,
            role="system",
            content=f"[MEMORY_SUMMARY] {summary}",
            model_used="memory_agent",
        )

        return {"content": summary, "source": "memory", "raw": formatted}


memory_agent = MemoryAgent()
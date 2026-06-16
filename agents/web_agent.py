from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.base import BaseAgent
from agents.config import AgentConfig
from agents.prompts import web_prompt
from tools.web.web_tools import search_web


class WebAgent(BaseAgent):
    """
    Web Agent: Searches the web for real-time information and
    summarizes the results using an LLM.

    Dependencies are injected via the constructor for testability
    and loose coupling.
    """

    def __init__(
        self,
        llm: ChatOpenAI = None,
        prompt: ChatPromptTemplate = None,
        search_fn=None,
    ):
        self.llm = llm or AgentConfig.web_llm()

        self.prompt = prompt or web_prompt()

        self.chain = self.prompt | self.llm | StrOutputParser()
        self._search_fn = search_fn or search_web

    @staticmethod
    def _normalize(result):
        if result is None:
            return ""

        if hasattr(result, "content"):
            result = result.content

        if isinstance(result, dict):
            result = result.get("content", "") or result.get("results", "")

        if isinstance(result, list):
            return "\n".join(
                str(x.get("content", x)) if isinstance(x, dict) else str(x)
                for x in result
            )

        return str(result)

    async def run(self, query: str):
        results = await self._search_fn(query)

        text = self._normalize(results)

        found = "no relevant" not in text.lower()

        summary = await self.chain.ainvoke({"query": query, "results": text})

        return {
            "content": summary,
            "source": "web",
            "raw": text,
            "found": found,
        }


web_agent = WebAgent()
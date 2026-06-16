from langchain_core.output_parsers import StrOutputParser
from tools.web.web_tools import search_web
from agents.config import AgentConfig
from agents.prompts import web_prompt

llm = AgentConfig.web_llm()

WEB_PROMPT = web_prompt()

web_chain = WEB_PROMPT | llm | StrOutputParser()


def normalize(result):
    if result is None:
        return ""

    if hasattr(result, "content"):
        result = result.content

    if isinstance(result, dict):
        result = result.get("content", "") or result.get("results", "")

    if isinstance(result, list):
        return "\n".join(
            str(x.get("content", x)) if isinstance(x, dict) else str(x) for x in result
        )

    return str(result)


async def run_web(query: str):

    results = await search_web(query)

    text = normalize(results)

    found = "no relevant" not in text.lower()

    summary = await web_chain.ainvoke({"query": query, "results": text})

    return {
        "content": summary,
        "source": "web",
        "raw": text,
        "found": found,
    }

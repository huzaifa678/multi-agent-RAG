from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from tools.web.web_tools import search_web
from utils.config import Config


llm = ChatOpenAI(
    api_key=Config.OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0
)


WEB_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract key factual insights from web results."),
    ("human",
     "Query: {query}\n\nResults:\n{results}")
])


web_chain = WEB_PROMPT | llm | StrOutputParser()


async def run_web(query: str):
    results = await search_web(query)

    summary = await web_chain.ainvoke({
        "query": query,
        "results": results
    })

    return {
        "content": summary,
        "source": "web",
        "raw": results
    }
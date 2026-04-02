from typing import List, TypedDict, Any

from langchain_tavily import TavilySearch
from langchain_community.retrievers.wikipedia import WikipediaRetriever
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

from utils.config import Config


tavily = TavilySearch(
    max_results=5,
    search_depth="advanced",
    api_key=Config.TAVILY_API_KEY
)


wiki_retriever = WikipediaRetriever(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=2000
    )
)


class SearchResult(TypedDict):
    source: str
    title: str
    url: str
    content: str


def web_search(query: str) -> List[SearchResult]:
    results: List[SearchResult] = []

    try:
        tavily_raw = tavily.invoke(query)

        tavily_results = (
            tavily_raw.get("results", [])
            if isinstance(tavily_raw, dict)
            else tavily_raw
        )

        for r in tavily_results:
            results.append({
                "source": "tavily",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")
            })

    except Exception as e:
        print(f"Tavily error: {e}")

    try:
        wiki_docs = wiki_retriever.invoke(query)

        for doc in wiki_docs:
            results.append({
                "source": "wikipedia",
                "title": doc.metadata.get("title", query),
                "url": doc.metadata.get("source", "") or doc.metadata.get("url", ""),
                "content": doc.page_content
            })

    except Exception as e:
        print(f"Wikipedia error: {e}")

    return results
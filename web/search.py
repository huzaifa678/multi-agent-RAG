import asyncio
from typing import List, TypedDict
from langchain_tavily import TavilySearch
from langchain_community.retrievers.wikipedia import WikipediaRetriever
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
import requests
from utils.config import Config
from langsmith import Client

tavily = TavilySearch(
    max_results=5, search_depth="advanced", api_key=Config.TAVILY_API_KEY
)

wiki_retriever = WikipediaRetriever(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
)

client = Client()


class SearchResult(TypedDict):
    source: str
    title: str
    url: str
    content: str


def trace_safe_results(results, content_limit=800):
    """
    Reduce payload ONLY for LangSmith tracing.
    Does NOT affect actual RAG output.
    """
    return [
        {
            "source": r.get("source"),
            "title": r.get("title", "")[:200],
            "url": r.get("url", ""),
            "content": r.get("content", "")[:content_limit],
        }
        for r in results
    ]


def fast_wikipedia(query: str):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        r = requests.get(url, timeout=2)

        if r.status_code != 200:
            return []

        data = r.json()

        return [
            {
                "source": "wikipedia",
                "title": data.get("title", ""),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "content": data.get("extract", ""),
            }
        ]

    except Exception as e:
        print("Wiki error:", e)
        return []


async def web_search(query: str) -> List[dict]:
    print("\n[WEB] START:", query)

    tavily_task = asyncio.create_task(asyncio.to_thread(tavily.invoke, query))

    wiki_task = asyncio.create_task(asyncio.to_thread(fast_wikipedia, query))

    try:
        done, pending = await asyncio.wait(
            [tavily_task, wiki_task], timeout=5, return_when=asyncio.FIRST_COMPLETED
        )
    except Exception as e:
        print("WAIT ERROR:", e)
        return []

    results = []

    for task in done:
        try:
            res = await task
        except Exception as e:
            print("Task failed:", e)
            continue

        if isinstance(res, dict):
            tavily_results = res.get("results", [])
            print(f"Tavily returned {len(tavily_results)} results")

            for r in tavily_results:
                results.append(
                    {
                        "source": "tavily",
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                    }
                )

        elif isinstance(res, list):
            print(f"Wiki returned {len(res)} results")
            results.extend(res)

    for task in pending:
        print("Cancelling slow task...")
        task.cancel()

    trace_output = trace_safe_results(results)

    client.create_run(
        name="web-search",
        inputs={"query": query},
        outputs={"results": trace_output},
        run_type="tool",
    )

    print(f"[WEB] DONE {len(results)} results\n")

    return results

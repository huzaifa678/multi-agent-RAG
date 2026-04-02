from tools.web.client import web_client


async def search_web(query: str):
    return await web_client.call_tool(
        "search_web",
        {"query": query}
    )
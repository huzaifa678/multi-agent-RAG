from tools.web.client import get_web_client


async def search_web(query: str):
    web_client = await get_web_client()
    return await web_client.call_tool("search_web", {"query": query})

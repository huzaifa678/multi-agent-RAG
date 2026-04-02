from tools.web.client import web_client


def search_web(query: str):
    return web_client.call_tool(
        "search_web",
        {"query": query}
    )
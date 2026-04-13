from fastmcp import FastMCP
from web.search import web_search

mcp = FastMCP("web-mcp-server")
app = mcp.http_app


@mcp.tool()
def search_web(query: str):
    """
    Hybrid web search (Tavily + Wikipedia)
    """
    return web_search(query)


if __name__ == "__main__":
    mcp.run()

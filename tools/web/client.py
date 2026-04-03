from fastmcp import Client
from utils.config import Config

web_client: Client | None = None


async def get_web_client():
    global web_client

    if web_client is None:
        web_client = Client(Config.MCP_WEB_URL)
        await web_client.__aenter__()

    return web_client


async def close_web_client():
    global web_client

    if not web_client:
        return

    try:
        if hasattr(web_client, "aclose"):
            await web_client.aclose()
        elif hasattr(web_client, "close"):
            await web_client.close()
        else:
            await web_client.__aexit__(None, None, None)
    finally:
        web_client = None
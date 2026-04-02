from fastmcp import Client

web_client: Client | None = None


async def get_web_client():
    global web_client

    if web_client is None:
        web_client = Client("http://localhost:8002")
        await web_client.__aenter__()

    return web_client


async def close_mcp():
    global web_client

    if web_client:
        await web_client.__aexit__(None, None, None)
        web_client = None
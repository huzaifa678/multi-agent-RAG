import os
from fastmcp import Client

memory_client: Client | None = None

memory_url = os.getenv("MCP_MEMORY_URL")

async def get_memory_client():
    global memory_client

    if memory_client is None:
        memory_client = Client(memory_url)
        await memory_client.__aenter__()

    return memory_client


async def close_memory_client():
    global memory_client

    if not memory_client:
        return

    try:
        if hasattr(memory_client, "aclose"):
            await memory_client.aclose()
        elif hasattr(memory_client, "close"):
            await memory_client.close()
        else:
            await memory_client.__aexit__(None, None, None)
    finally:
        memory_client = None
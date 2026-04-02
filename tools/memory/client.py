from fastmcp import Client

memory_client: Client | None = None


async def get_memory_client():
    global memory_client

    if memory_client is None:
        memory_client = Client("http://localhost:8002/mcp")
        await memory_client.__aenter__()

    return memory_client


async def close_mcp():
    global memory_client

    if memory_client:
        await memory_client.__aexit__(None, None, None)
        memory_client = None
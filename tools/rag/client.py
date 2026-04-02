from fastmcp import Client

rag_client: Client | None = None


async def get_rag_client():
    global rag_client

    if rag_client is None:
        rag_client = Client("http://localhost:8001/mcp")
        await rag_client.__aenter__()

    return rag_client


async def close_mcp():
    global rag_client

    if rag_client:
        await rag_client.__aexit__(None, None, None)
        rag_client = None
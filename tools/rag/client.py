from fastmcp import Client
from utils.config import Config


rag_client: Client | None = None

async def get_rag_client():
    global rag_client

    if rag_client is None:
        rag_client = Client(Config.MCP_RAG_URL)
        await rag_client.__aenter__()

    return rag_client


async def close_rag_client():
    global rag_client

    if not rag_client:
        return

    try:
        if hasattr(rag_client, "aclose"):
            await rag_client.aclose()
        elif hasattr(rag_client, "close"):
            await rag_client.close()
        else:
            await rag_client.__aexit__(None, None, None)
    finally:
        rag_client = None
from fastmcp import Client

rag_client = Client("http://localhost:8001")

async def init_mcp():
    await rag_client.__aenter__()

async def close_mcp():
    await rag_client.__aexit__(None, None, None)
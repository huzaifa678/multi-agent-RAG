from tools.rag.client import get_rag_client, rag_client
from tools.rag.wrapper import rag_client_wrapper


async def retrieve_documents(query: str, top_k: int = 4):
    return await rag_client_wrapper.retrieve_documents(query, top_k)


async def add_documents(docs: list):
    client = await get_rag_client()

    return await client.call_tool("add_documents_tool", {"docs": docs})

import asyncio

from tools.rag.client import rag_client

async def retrieve_documents(query: str, top_k: int = 4):
    return await rag_client.call_tool(
        "retrieve_documents_tool",
        {"query": query, "top_k": top_k}
    )

def add_documents(docs: list):
    return rag_client.call_tool(
        "add_documents_tool",
        {
            "docs": docs
        }
    )
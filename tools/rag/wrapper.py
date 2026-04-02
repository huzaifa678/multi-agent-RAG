from tools.rag.client import get_rag_client

class RAGClientWrapper:
    async def retrieve_documents(self, query: str, top_k: int = 4):
        client = await get_rag_client()
        return await client.call_tool(
            "retrieve_documents_tool",
            {"query": query, "top_k": top_k}
        )

rag_client_wrapper = RAGClientWrapper()
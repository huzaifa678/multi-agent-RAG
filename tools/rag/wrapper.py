from tools.rag.client import rag_client

class RAGClientWrapper:
    def __init__(self, client):
        self.client = client

    async def retrieve_documents(self, query: str, top_k: int = 4):
        async with self.client:
            return await self.client.call_tool(
                "retrieve_documents_tool",
                {"query": query, "top_k": top_k}
            )

rag_client_wrapper = RAGClientWrapper(rag_client)
from fastmcp import FastMCP
from rag.add_documents import add_documents
from rag.retriever import retrieve_context as retrieve_documents

mcp = FastMCP("rag-mcp-server")


@mcp.tool()
def add_documents_tool(docs: list):
    """
    Add documents to RAG vectorstore
    """
    return add_documents(docs)

@mcp.tool()
def retrieve_documents_tool(query: str, top_k: int = 5):
    """
    Retrieve relevant documents from vectorstore
    """
    return retrieve_documents(query, top_k)


if __name__ == "__main__":
    mcp.run()
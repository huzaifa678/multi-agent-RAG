from rag.chroma_store import vectorstore

def retrieve_context(query: str, k: int = 4):
    results = vectorstore.similarity_search(query, k=k)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in results
    ]
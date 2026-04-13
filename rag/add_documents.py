from rag.chroma_store import vectorstore


def add_documents(docs: list):
    texts = [d["text"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]

    vectorstore.add_texts(texts=texts, metadatas=metadatas)

    return {"status": "success", "added": len(texts)}

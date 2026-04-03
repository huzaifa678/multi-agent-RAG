from utils.text import chunk_text, extract_text
from rag.chroma_store import vectorstore

async def process_document(file_id: str, file_path: str):
    text = extract_text(file_path)
    chunks = chunk_text(text)

    texts = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        texts.append(chunk)
        metadatas.append({
            "file_id": file_id,
            "chunk_id": i,
            "source": file_path
        })

    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas
    )
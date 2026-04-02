from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pymupdf import Document
from utils.config import Config
from utils.text import chunk_text


embedding_function = OpenAIEmbeddings(
    openai_api_key=Config.OPENAI_API_KEY
)

vectorstore = Chroma(
    persist_directory=Config.CHROMA_PATH,
    embedding_function=embedding_function
)

async def write_to_chroma(query: str, content: str):
    doc = Document(
        page_content=content,
        metadata={
            "source": "auto_update",
            "query": query
        }
    )

    chunks = chunk_text().split_documents([doc])

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    vectorstore.add_texts(texts=texts, metadatas=metadatas)
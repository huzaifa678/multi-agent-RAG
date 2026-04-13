from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

from utils.logger import get_logger

logger = get_logger()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, length_function=len
)


def load_and_split_documents(file_path: str):
    """
    Load + split documents into chunks for RAG ingestion
    """

    if file_path.endswith(".txt"):
        loader = TextLoader(file_path)

    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)

    else:
        raise ValueError("Unsupported file format (.txt, .pdf, .docx only)")

    documents = loader.load()
    chunks = text_splitter.split_documents(documents)

    for i, doc in enumerate(chunks):
        doc.metadata.update({"source": file_path, "chunk_id": i})

    logger.info(f"Loaded {len(chunks)} chunks from {file_path}")

    return chunks

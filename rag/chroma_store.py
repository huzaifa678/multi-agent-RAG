from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from utils.config import Config

embedding_function = OpenAIEmbeddings(
    openai_api_key=Config.OPENAI_API_KEY
)

vectorstore = Chroma(
    persist_directory=Config.CHROMA_PATH,
    embedding_function=embedding_function
)
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    CHROMA_PATH = os.getenv("CHROMA_PATH")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
    MCP_RAG_URL = os.getenv("MCP_RAG_URL")
    MCP_WEB_URL = os.getenv("MCP_WEB_URL")
    MCP_MEMORY_URL = os.getenv("MCP_MEMORY_URL")
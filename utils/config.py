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
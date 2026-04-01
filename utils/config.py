import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    CHROMA_PATH = os.getenv("CHROMA_PATH")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
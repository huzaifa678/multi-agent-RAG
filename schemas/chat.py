from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    query: str
    session_id: str
    history: Optional[List[str]] = []

from typing import Optional
from tools.memory.client import get_memory_client

async def save_message(session_id: str, role: str, content: str, model_used: Optional[str] = None):
    memory_client = await get_memory_client()
    return await memory_client.call_tool(
        "save_message",
        {
            "session_id": session_id,
            "role": role,
            "content": content,
            "model_used": model_used
        }
    )
    

async def get_history(session_id: str, limit: int = 10):
    memory_client = await get_memory_client()
    return await memory_client.call_tool(
        "get_history",
        {
            "session_id": session_id,
            "limit": limit
        }
    )
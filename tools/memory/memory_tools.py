from tools.memory.client import memory_client

async def save_message(session_id: str, role: str, content: str, model_used: str = None):
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
    return await memory_client.call_tool(
        "get_history",
        {
            "session_id": session_id,
            "limit": limit
        }
    )
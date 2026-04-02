from tools.memory.client import memory_client

def save_message(session_id: str, role: str, content: str, model_used: str = None):
    return memory_client.call_tool(
        "save_message",
        {
            "session_id": session_id,
            "role": role,
            "content": content,
            "model_used": model_used
        }
    )


def get_history(session_id: str, limit: int = 10):
    return memory_client.call_tool(
        "get_history",
        {
            "session_id": session_id,
            "limit": limit
        }
    )
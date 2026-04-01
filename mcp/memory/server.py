from fastmcp import FastMCP
from memory.sqllite_memory import (
    insert_message,
    get_chat_history
)

mcp = FastMCP("memory-mcp-server")


@mcp.tool()
def save_message(session_id: str, role: str, content: str, model_used: str = None):
    """
    Store chat message into SQLite memory.
    """
    insert_message(session_id, role, content, model_used)
    return {"status": "saved"}


@mcp.tool()
def get_history(session_id: str, limit: int = 10):
    """
    Retrieve chat history for a session.
    """
    return get_chat_history(session_id, limit)


if __name__ == "__main__":
    mcp.run()
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils.config import Config
from tools.memory.memory_tools import get_history
from tools.memory.memory_tools import save_message
from utils.logger import get_logger

logger = get_logger()

llm = ChatOpenAI(api_key=Config.OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)

MEMORY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Analyze the provided conversation history.
1. If you find user preferences, goals, or personal context, list them as 'Stable Facts'.
2. If the conversation is a general topic (like Rust programming), provide a 1-sentence summary of the discussion topic.
3. Combine these into a concise memory entry.
4. If the history is completely empty or contains no meaningful information, return: NO_MEMORY
""",
        ),
        ("human", "{history}"),
    ]
)

memory_chain = MEMORY_PROMPT | llm | StrOutputParser()



async def run_memory(session_id: str):

    raw = await get_history(session_id)

    history_list = []

    if hasattr(raw, "data") and isinstance(raw.data, dict):
        logger.info("The attribute is data")
        history_list = raw.data.get("structured_content") or []

    if not history_list or not isinstance(history_list, list):
        logger.warning("No content to use the memory is empty")
        return {"content": "NO_MEMORY", "source": "memory", "raw": ""}

    valid_items = [h for h in history_list if isinstance(h, dict) and h.get("content")]

    formatted = "\n".join(
        f"{h.get('role','unknown')}: {h.get('content','')}" for h in valid_items
    )

    if not formatted.strip():
        return {"content": "NO_MEMORY", "source": "memory", "raw": ""}

    summary = await memory_chain.ainvoke({"history": formatted})

    await save_message(
        session_id,
        role="system",
        content=f"[MEMORY_SUMMARY] {summary}",
        model_used="memory_agent",
    )

    return {"content": summary, "source": "memory", "raw": formatted}

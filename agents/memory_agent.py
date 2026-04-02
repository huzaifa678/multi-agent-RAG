from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tools.memory.memory_tools import get_history
from utils.config import Config

llm = ChatOpenAI(
    api_key=Config.OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0
)

MEMORY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Summarize chat history into compact factual memory. Remove noise."),
    ("human",
     "{history}")
])

memory_chain = MEMORY_PROMPT | llm | StrOutputParser()

async def run_memory(session_id: str):
    history = await get_history(session_id)

    summary = await memory_chain.ainvoke({
        "history": history
    })

    return {
        "content": summary,
        "source": "memory",
        "raw": history
    }
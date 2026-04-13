import asyncio
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from utils.config import Config

SYSTEM = """
You are a query decomposition engine for an AI agent system.

Convert the user input into a standalone optimized query.

Rules:
- Preserve intent
- Resolve references using chat history
- Make query tool-ready (RAG/Web friendly)
- Do NOT answer
- Output ONLY rewritten query
"""


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


groq = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0, api_key=Config.GROQ_API_KEY
)

openai = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=Config.OPENAI_API_KEY)


def build_chain(llm):
    return PROMPT | llm | StrOutputParser()


groq_chain = build_chain(groq)
openai_chain = build_chain(openai)


async def contextualize(state: dict) -> str:
    history = state.get("history", [])

    try:
        return await asyncio.to_thread(
            groq_chain.invoke, {"input": state["input"], "history": history}
        )

    except Exception:
        return await asyncio.to_thread(
            openai_chain.invoke, {"input": state["input"], "history": history}
        )

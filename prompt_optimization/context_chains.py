from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from utils.config import Config


contextualize_system_prompt = (
    "You are a query rewriting engine.\n"
    "Your job is to convert the user's latest message into a standalone question.\n\n"
    "Rules:\n"
    "- Use chat history only for context\n"
    "- Do NOT answer the question\n"
    "- If already standalone, return as-is\n"
    "- Output ONLY the rewritten query\n"
)


CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", contextualize_system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=Config.GROQ_API_KEY
)

openai_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=Config.OPENAI_API_KEY
)


def _build_chain(llm):
    return (
        CONTEXT_PROMPT
        | llm
        | StrOutputParser()
    )


def contextualize_chain(state: dict) -> str:
    """
    Smart fallback:
    1. Try Groq (fast)
    2. Fallback to OpenAI
    """

    chain = _build_chain(groq_llm)

    try:
        return chain.invoke({
            "input": state["input"],
            "history": state.get("history", [])
        })
    except Exception:
        fallback_chain = _build_chain(openai_llm)
        return fallback_chain.invoke({
            "input": state["input"],
            "history": state.get("history", [])
        })
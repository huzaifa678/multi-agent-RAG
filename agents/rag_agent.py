import asyncio

from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from utils.text import chunk_text
from rag.retriever import retrieve_context
from rag.chroma_store import vectorstore
from utils.text import clean_text
from agents.config import AgentConfig
from agents.prompts import rag_prompt

llm = AgentConfig.rag_llm()

RAG_PROMPT = rag_prompt()

rag_chain = RAG_PROMPT | llm | StrOutputParser()


@traceable(name="rag_agent_with_update")
async def run_rag(query: str, web_context: str = None):

    result = retrieve_context(query)

    found = bool(result and len(result) > 0)

    if not found and web_context and "no relevant" not in web_context.lower():
        print(f"RAG Agent: Learning from web context for query: {query}")
        cleaned = clean_text(web_context)
        chunks = chunk_text(cleaned)

        asyncio.create_task(
            asyncio.to_thread(
                vectorstore.add_texts,
                texts=chunks,
                metadatas=[{"source": "auto_update", "query": query}] * len(chunks),
            )
        )

        result = retrieve_context(query)
        found = bool(result and len(result) > 0)

    context = "\n".join([d["content"] for d in result]) if found else "NOT_FOUND"

    answer = await rag_chain.ainvoke({"query": query, "context": context})

    return {"content": answer, "source": "rag", "found": found}

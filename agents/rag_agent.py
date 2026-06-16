from __future__ import annotations

import asyncio

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from agents.base import BaseAgent
from agents.config import AgentConfig
from agents.prompts import rag_prompt
from rag.chroma_store import vectorstore
from rag.retriever import retrieve_context
from utils.text import chunk_text, clean_text


class RAGAgent(BaseAgent):
    """
    RAG Agent: Retrieves context from the vector store and generates
    answers using an LLM. Optionally ingests web context into the store.

    Dependencies are injected via the constructor for testability
    and loose coupling.
    """

    def __init__(
        self,
        llm: ChatGroq = None,
        prompt: ChatPromptTemplate = None,
        retrieve_fn=None,
        vectorstore_ref=None,
    ):
        self.llm = llm or AgentConfig.rag_llm()

        self.prompt = prompt or rag_prompt()

        self.chain = self.prompt | self.llm | StrOutputParser()
        self._retrieve_fn = retrieve_fn or retrieve_context
        self._vectorstore = vectorstore_ref or vectorstore

    @traceable(name="rag_agent_with_update")
    async def run(self, query: str, web_context: str = None):
        result = self._retrieve_fn(query)

        found = bool(result and len(result) > 0)

        if not found and web_context and "no relevant" not in web_context.lower():
            print(f"RAG Agent: Learning from web context for query: {query}")
            cleaned = clean_text(web_context)
            chunks = chunk_text(cleaned)

            asyncio.create_task(
                asyncio.to_thread(
                    self._vectorstore.add_texts,
                    texts=chunks,
                    metadatas=[{"source": "auto_update", "query": query}]
                    * len(chunks),
                )
            )

            result = self._retrieve_fn(query)
            found = bool(result and len(result) > 0)

        context = (
            "\n".join([d["content"] for d in result]) if found else "NOT_FOUND"
        )

        answer = await self.chain.ainvoke({"query": query, "context": context})

        return {"content": answer, "source": "rag", "found": found}


rag_agent = RAGAgent()
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from tools.rag.rag_tools import retrieve_documents
from utils.config import Config


llm = ChatGroq(
    api_key=Config.GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0
)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer ONLY using context. If missing, say 'not found in knowledge base'."),
    ("human",
     "Query: {query}\n\nContext:\n{context}")
])


rag_chain = RAG_PROMPT | llm | StrOutputParser()


@traceable(name="rag_retrieval")
def run_rag(query: str):
    docs = retrieve_documents(query)

    context = "\n".join([d["content"] for d in docs])

    answer = rag_chain.invoke({
        "query": query,
        "context": context
    })

    return {
        "content": answer,
        "source": "rag",
        "raw": docs
    }
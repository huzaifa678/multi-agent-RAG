# Multi-Agent RAG System

A production-grade multi-agent Retrieval-Augmented Generation (RAG) system built with LangGraph, FastAPI, and Model Context Protocol (MCP). The system orchestrates multiple specialized AI agents — RAG, Web Search, and Memory — through a dynamic planning and replanning workflow to answer user queries with high accuracy.

---

## Architecture Overview

```
User Query
    │
    ▼
[Chat Service] ──► Query Contextualization (Groq / OpenAI fallback)
    │
    ▼
[LangGraph Workflow]
    │
    ├──► [Planner Node]        — Decides which agents to call + confidence scores
    │
    ├──► [Memory Agent]        — Retrieves session history via MCP Memory Server
    ├──► [RAG Agent]           — Semantic search via ChromaDB (Groq LLaMA 3.3 70B)
    ├──► [Web Agent]           — Tavily + Wikipedia search (GPT-4o-mini)
    │
    ├──► [Replan Node]         — ReAct evaluator: done or need more tools?
    │
    └──► [Aggregator Node]     — Final synthesis (Claude Sonnet + GPT-4o-mini fallback)
```

### MCP Microservices

Each agent communicates with its own MCP (Model Context Protocol) server over HTTP:

| Service       | Port | Responsibility                        |
|---------------|------|---------------------------------------|
| Backend API   | 8000 | FastAPI — chat, upload, health        |
| MCP RAG       | 8001 | ChromaDB vector store operations      |
| MCP Web       | 8002 | Tavily + Wikipedia search             |
| MCP Memory    | 8003 | SQLite conversation memory            |
| ChromaDB      | 8004 | Vector database                       |

---

## Tech Stack

| Layer            | Technology                                      |
|------------------|-------------------------------------------------|
| Orchestration    | LangGraph (StateGraph with conditional routing) |
| LLM — Planning   | Claude Sonnet (Anthropic)                       |
| LLM — RAG        | LLaMA 3.3 70B (Groq)                           |
| LLM — Web/Memory | GPT-4o-mini (OpenAI)                           |
| LLM — Fallback   | GPT-4o-mini (OpenAI)                           |
| Vector Store     | ChromaDB + OpenAI Embeddings                   |
| Memory           | SQLite (short-term + long-term)                |
| Web Search       | Tavily + Wikipedia REST API                    |
| Observability    | LangSmith tracing                              |
| API              | FastAPI + Uvicorn                              |
| Frontend         | Vue 3 + Vite + TypeScript                      |
| Containerization | Docker + Docker Compose                        |
| Protocol         | MCP (Model Context Protocol) via FastMCP       |

---

## Project Structure

```
Multi-Agent-RAG/
├── main.py                        # FastAPI app entry point
├── worker.py                      # Background document ingestion worker
├── docker-compose.yml
├── requirements.txt
│
├── api/
│   ├── chat.py                    # POST /chat endpoint
│   ├── upload.py                  # POST /upload-doc endpoint
│   └── health.py                  # GET /health endpoint
│
├── graph/
│   └── workflow.py                # LangGraph StateGraph definition + execution
│
├── agents/
│   ├── aggregator_agent.py        # Final synthesis + memory persistence
│   ├── rag_agent.py               # ChromaDB retrieval + auto-learning from web
│   ├── web_agent.py               # Web search summarization
│   └── memory_agent.py            # Session memory analysis
│
├── rag/
│   ├── loader.py                  # PDF / DOCX / TXT document loader + splitter
│   ├── chroma_store.py            # ChromaDB client + vectorstore setup
│   ├── retriever.py               # Similarity search wrapper
│   └── add_documents.py           # Manual document ingestion utility
│
├── memory/
│   └── sqllite_memory.py          # SQLite chat history + long-term memory
│
├── mcp_servers/
│   ├── rag/server.py              # MCP RAG server (port 8001)
│   ├── web/server.py              # MCP Web server (port 8002)
│   └── memory/server.py           # MCP Memory server (port 8003)
│
├── tools/
│   ├── rag/                       # RAG MCP client + tools
│   ├── web/                       # Web MCP client + tools
│   └── memory/                    # Memory MCP client + tools
│
├── services/
│   ├── chat_service.py            # Chat orchestration service
│   └── upload_service.py          # File upload + background ingestion
│
├── schemas/
│   ├── plan.py                    # PlanSchema (Pydantic)
│   ├── replan.py                  # ReplanSchema (Pydantic)
│   └── chat.py                    # ChatRequest schema
│
├── prompt_optimization/
│   └── context_chains.py          # Query contextualization / rewriting
│
├── web/
│   └── search.py                  # Tavily + Wikipedia search logic
│
├── core/
│   └── runtime.py                 # MCP client lifecycle management
│
├── utils/
│   ├── config.py                  # Environment config
│   ├── logger.py                  # Logging setup
│   ├── text.py                    # Text chunking + extraction utilities
│   ├── sync.py                    # Async/sync helpers
│   └── timing.py                  # Latency measurement
│
└── frontend/                      # Vue 3 chat UI
```

---

## How It Works

### Query Flow

1. **Contextualization** — The raw user query is rewritten into a standalone, tool-ready query using Groq LLaMA (with OpenAI fallback).
2. **Planning** — The planner node uses Claude Sonnet to decide which agents to invoke (RAG, Web, Memory) with confidence scores. Memory is always included.
3. **Agent Execution** — Agents run in a dynamic order determined by `route_tools`. Memory runs first, then RAG and Web as needed.
4. **Replanning (ReAct loop)** — After each agent runs, the replan node evaluates whether the gathered information is sufficient or if more tools are needed.
5. **RAG Auto-Learning** — If RAG returns "not found" but Web has results, the web content is automatically chunked and stored in ChromaDB for future queries.
6. **Aggregation** — Claude Sonnet synthesizes the final answer from all sources. GPT-4o-mini is used as a fallback if Anthropic fails.
7. **Memory Persistence** — The Q&A pair is asynchronously saved to long-term memory (SQLite).

### Document Ingestion

Upload a document via `POST /upload-doc`. The file is saved locally and processed in the background:
- Text is extracted from PDF / DOCX / TXT
- Split into 1000-character chunks with 200-character overlap
- Embedded via OpenAI Embeddings and stored in ChromaDB

---

## Setup

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Node.js + pnpm (for frontend)

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...

LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=multi-agent-rag
LANGCHAIN_TRACING_V2=true

CHROMA_PATH=./chroma_db          # local dev
CHROMA_HOST=chroma               # docker
CHROMA_PORT=8000
COLLECTION_NAME=documents

MCP_RAG_URL=http://localhost:8001/mcp
MCP_WEB_URL=http://localhost:8002/mcp
MCP_MEMORY_URL=http://localhost:8003/mcp

ENVIRONMENT=local                # or "docker"
```

### Run with Docker (Recommended)

```bash
docker compose up --build
```

### Run Locally

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start MCP servers (in separate terminals)
python -m mcp_servers.rag.server
python -m mcp_servers.web.server
python -m mcp_servers.memory.server

# Start backend
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
pnpm install
pnpm run dev
```

---

## API Reference

### `POST /chat`

```json
{
  "query": "What is the capital of France?",
  "session_id": "user-123",
  "history": []
}
```

**Response:**
```json
{
  "response": "The capital of France is Paris.",
  "debug": {
    "rag": "...",
    "web": "...",
    "memory": "...",
    "agent_calls": ["memory", "web"],
    "executed_calls": ["memory", "web"]
  }
}
```

### `POST /upload-doc`

Multipart form upload. Accepts `.pdf`, `.docx`, `.txt`.

```bash
curl -X POST http://localhost:8000/upload-doc \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "file_id": "uuid",
  "status": "processing"
}
```

### `GET /health`

Returns service health status.

---

## Observability

All major operations are traced via LangSmith:
- `chat_service` — full chat pipeline
- `langgraph_workflow` — graph execution
- `rag_agent_with_update` — RAG retrieval + auto-learning
- `upload_service` / `upload_doc_endpoint` — document ingestion

Set `LANGCHAIN_TRACING_V2=true` and provide `LANGCHAIN_API_KEY` to enable.

---

## Known Issues & Possible Improvements

### LangSmith Trace Payload Limit (Document Ingestion)

When uploading large documents (e.g. PDFs > ~100 pages), the `@traceable` decorator on `upload_service` can cause the trace to **pause or fail** because LangSmith has a payload size limit per run (~1MB). The full document content gets captured in the trace inputs, exceeding this limit. This issue is also seen in web_search where the retrieved content from the web can possibly exceed the LangSmith payload size limit.

**Current workaround:** `web/search.py` already implements `trace_safe_results()` which truncates content before sending to LangSmith. The same pattern needs to be applied to the upload pipeline — stripping raw document content from trace inputs and only logging metadata (file_id, filename, chunk count).

**Fix (planned):**
```python
# In upload_service.py — pass only metadata to the traceable context
# instead of the full file content
```

### Latency

| Stage         | Before | After optimization |
|---------------|--------|--------------------|
| End-to-end    | 70–80s | 20–30s             |

Improvements made:
- Web search now uses `asyncio.wait(..., return_when=FIRST_COMPLETED)` with a 5-second timeout, cancelling slow tasks
- Wikipedia switched from LangChain retriever to a direct REST API call (`/api/rest_v1/page/summary`) — eliminates LangChain overhead
- Memory and DB writes are fire-and-forget via `asyncio.create_task`
- Context passed to the final LLM is trimmed to `MAX_CONTEXT_CHARS` to reduce token processing time

**Still needed:**
- RAG retrieval is synchronous (`similarity_search`) — wrapping in `asyncio.to_thread` would unblock the event loop
- The planner and replan nodes make sequential LLM calls — these could be parallelized where agent results are independent
- ChromaDB cold-start on first query adds ~2–5s — connection pooling or a warm-up ping at startup would help
- Streaming responses (`StreamingResponse`) would improve perceived latency significantly even before total time is reduced

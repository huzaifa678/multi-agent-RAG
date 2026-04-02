from fastapi import FastAPI
from contextlib import asynccontextmanager
from ragas.cli import app
from api import health, upload, chat
from core import runtime
from graph.workflow import build_workflow_graph
from mcp_servers.rag import shutdown_handler
from tools.rag.client import close_mcp, close_mcp
from tools.rag.client import get_rag_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    runtime.app_graph = build_workflow_graph()  # Build the graph once at startup
    yield
    runtime.app_graph = None

app = FastAPI(lifespan=lifespan)

@app.on_event("startup")
async def startup():
    await get_rag_client()


@app.on_event("shutdown")
async def shutdown():
    await close_mcp()

@app.on_event("shutdown")
async def shutdown_event():
    await shutdown_handler(timeout=10)


app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(health.router)
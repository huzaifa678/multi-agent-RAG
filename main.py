from fastapi import FastAPI
from contextlib import asynccontextmanager
from api import health, upload, chat
from core import runtime
from graph.workflow import build_workflow_graph
from core.runtime import runtimeObject


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.app_graph = build_workflow_graph()  # Build the graph once at startup
    await runtimeObject.init()
    yield
    await runtimeObject.shutdown()


app = FastAPI(lifespan=lifespan)

app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(health.router)

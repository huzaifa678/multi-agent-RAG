from fastapi import FastAPI
from contextlib import asynccontextmanager

from graph.workflow import build_workflow_graph

app_graph = None  


@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_graph
    app_graph = build_workflow_graph()  # Build the graph once at startup
    yield
    app_graph = None


app = FastAPI(lifespan=lifespan)
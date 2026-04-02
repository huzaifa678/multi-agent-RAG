from fastapi import FastAPI
from contextlib import asynccontextmanager


from api import health, upload, chat
from graph.workflow import build_workflow_graph

app_graph = None  

@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_graph
    app_graph = build_workflow_graph()  # Build the graph once at startup
    yield
    app_graph = None


app = FastAPI(lifespan=lifespan)

app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(health.router)
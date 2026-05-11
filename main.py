from fastapi import FastAPI
from contextlib import asynccontextmanager
from api import health, upload, chat
from core import runtime
from graph.workflow import Workflow
from graph import set_workflow
from core.runtime import runtimeObject


@asynccontextmanager
async def lifespan(app: FastAPI):
    workflow = Workflow()  # Build the workflow once at startup
    set_workflow(workflow)  # Register the shared instance
    runtime.app_graph = workflow
    await runtimeObject.init()
    yield
    await runtimeObject.shutdown()


app = FastAPI(lifespan=lifespan)

app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(health.router)
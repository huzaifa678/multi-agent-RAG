import asyncio

from fastapi import APIRouter

from prompt_optimization.context_chains import contextualize
from graph.workflow import execute_workflow
import router  

@router.post("/chat")
async def chat(payload: dict):
    rewritten_query = contextualize({
        "input": payload["query"],
        "history": payload.get("history", [])
    })

    result = await asyncio.to_thread(
        execute_workflow,
        rewritten_query,
        payload["session_id"]
    )

    return result
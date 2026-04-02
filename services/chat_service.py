import asyncio
from graph.workflow import execute_workflow
from prompt_optimization.context_chains import contextualize


async def handle_chat(payload: dict):

    query = payload["query"]
    session_id = payload["session_id"]

    rewritten_query = contextualize({
        "input": query,
        "history": payload.get("history", [])
    })

    result = await asyncio.to_thread(
        execute_workflow,
        rewritten_query,
        session_id
    )

    return result
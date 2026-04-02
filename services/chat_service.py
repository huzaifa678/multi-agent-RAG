import asyncio

from langsmith import traceable
from graph.workflow import execute_workflow
from prompt_optimization.context_chains import contextualize
from utils import logger

@traceable(name="chat_service")
async def handle_chat(payload: dict):
    try:
        query = payload["query"]
        session_id = payload["session_id"]

        logger.info(f"Chat service started | session_id={session_id} | query={query}")

        rewritten_query = await contextualize({
            "input": query,
            "history": payload.get("history", [])
        })

        logger.info(f"Query contextualized | session_id={session_id}")

        result = await asyncio.to_thread(
            execute_workflow,
            rewritten_query,
            session_id
        )

        logger.info(f"Workflow executed successfully | session_id={session_id}")

        return result

    except KeyError as e:
        logger.warning(f"Missing required field in payload: {str(e)}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in chat service | session_id={payload.get('session_id')}")

        raise
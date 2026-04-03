from langsmith import traceable
from graph.workflow import execute_workflow
from prompt_optimization.context_chains import contextualize
from schemas.chat import ChatRequest
from utils.logger import get_logger


logger = get_logger("chat-service")

@traceable(name="chat_service")
async def handle_chat(payload: ChatRequest):
    try:
        query = payload.query
        session_id = payload.session_id

        logger.info(f"Chat service started | session_id={session_id} | query={query}")

        rewritten_query = await contextualize({
            "input": query,
            "history": payload.history
        })

        logger.info(f"Query contextualized | session_id={session_id}")

        result = await execute_workflow(rewritten_query, session_id)

        logger.info(f"Workflow executed successfully | session_id={session_id}")

        return result

    except KeyError as e:
        logger.warning(f"Missing required field in payload: {str(e)}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in chat service | session_id={payload.session_id}")

        raise
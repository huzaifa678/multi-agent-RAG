from fastapi import APIRouter, HTTPException, Depends
from langsmith import traceable
from schemas.chat import ChatRequest
from services.chat_service import handle_chat
from graph import get_workflow_async
from utils.logger import get_logger

router = APIRouter()
logger = get_logger("chat-api")


@router.post("/chat")
@traceable(name="chat_endpoint")
async def chat(
    payload: ChatRequest,
    workflow=Depends(get_workflow_async),
):

    try:
        logger.info(f"Incoming chat request: {payload}")

        result = await handle_chat(payload, workflow=workflow)

        logger.info("Chat request processed successfully")
        return result

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unexpected error in /chat endpoint")

        raise HTTPException(status_code=500, detail="Internal Server Error")
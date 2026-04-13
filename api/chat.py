from fastapi import APIRouter, HTTPException
from langsmith import traceable
from schemas.chat import ChatRequest
from services.chat_service import handle_chat
from utils.logger import get_logger
from core.runtime import runtimeObject

router = APIRouter()
logger = get_logger("chat-api")


@router.post("/chat")
@traceable(name="chat_endpoint")
async def chat(payload: ChatRequest):

    if not runtimeObject.ready:
        raise HTTPException(status_code=503, detail="Service warming up")

    try:
        logger.info(f"Incoming chat request: {payload}")

        result = await handle_chat(payload)

        logger.info("Chat request processed successfully")
        return result

    except HTTPException as e:
        logger.warning(f"HTTPException occurred: {e.detail}")
        raise e

    except Exception:
        logger.exception("Unexpected error in /chat endpoint")

        raise HTTPException(status_code=500, detail="Internal Server Error")

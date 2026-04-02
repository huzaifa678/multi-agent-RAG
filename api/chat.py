from fastapi import APIRouter, HTTPException
from langsmith import traceable
from langsmith import traceable
from services.chat_service import handle_chat
from utils import logger

router = APIRouter()

@router.post("/chat")
@traceable(name="chat_endpoint")
async def chat(payload: dict):
    try:
        logger.info(f"Incoming chat request: {payload}")

        result = await handle_chat(payload)

        logger.info("Chat request processed successfully")
        return result

    except HTTPException as e:
        logger.warning(f"HTTPException occurred: {e.detail}")
        raise e

    except Exception as e:
        logger.exception("Unexpected error in /chat endpoint")

        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )
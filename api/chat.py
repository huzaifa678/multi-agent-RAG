from fastapi import APIRouter
from prompt_optimization.context_chains import contextualize
from services.chat_service import handle_chat

router = APIRouter()

@router.post("/chat")
async def chat(payload: dict):
    return await handle_chat(payload)
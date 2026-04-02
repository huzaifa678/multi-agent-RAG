from fastapi import APIRouter
from langsmith import traceable
from langsmith import traceable
from services.chat_service import handle_chat

router = APIRouter()

@router.post("/chat")
@traceable(name="chat_endpoint")
async def chat(payload: dict):
   
    return await handle_chat(payload)
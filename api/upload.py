from fastapi import APIRouter, File , UploadFile
import os
from services.upload_service import handle_upload

router = APIRouter()

@router.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...)):

    return await handle_upload(file)
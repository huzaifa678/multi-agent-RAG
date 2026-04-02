import uuid
from fastapi import APIRouter, File , UploadFile
import os
from worker import process_document

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

import asyncio

@router.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    content = await file.read()

    file_id = str(uuid.uuid4())
    file_path = f"uploads/{file_id}_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(content)

    # async ingestion
    asyncio.create_task(process_document(file_id, file_path))

    return {"file_id": file_id, "status": "processing"}
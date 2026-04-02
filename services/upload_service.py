import uuid
import os
import asyncio
from fastapi import UploadFile
from worker import process_document

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def handle_upload(file: UploadFile):

    content = await file.read()

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(file_path, "wb") as f:
        f.write(content)

    # background ingestion
    asyncio.create_task(process_document(file_id, file_path))

    return {
        "file_id": file_id,
        "status": "processing"
    }
import uuid
import os
import asyncio
from fastapi import HTTPException, UploadFile
from langsmith import traceable
from utils.logger import get_logger
from worker import process_document

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logger = get_logger("upload-service")

@traceable(name="upload_service")
async def handle_upload(file: UploadFile):
    try:
        logger.info(f"Upload started | filename={file.filename}")

        content = await file.read()

        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"File saved locally | file_id={file_id} | path={file_path}")

        # background ingestion
        task = asyncio.create_task(process_document(file_id, file_path))
        task.add_done_callback(
            lambda t: logger.info(f"Document processing finished | file_id={file_id}")
        )

        logger.info(f"Background ingestion started | file_id={file_id}")

        return {
            "file_id": file_id,
            "status": "processing"
        }

    except Exception as e:
        logger.exception(f"Upload failed | filename={getattr(file, 'filename', None)}")

        raise HTTPException(
            status_code=500,
            detail="File upload failed"
        )
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from langsmith import traceable
from services.upload_service import handle_upload
from utils.logger import get_logger

logger = get_logger("upload-api")
router = APIRouter()


@router.post("/upload-doc")
@traceable(name="upload_doc_endpoint")
async def upload_doc(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        logger.info(
            f"Upload request received: filename={file.filename}, content_type={file.content_type}"
        )

        result = await handle_upload(file, background_tasks)

        logger.info(f"Upload successful: filename={file.filename}")
        return result

    except HTTPException as e:
        logger.warning(f"HTTPException during upload: {e.detail}")
        raise e

    except Exception as e:
        logger.exception(f"Unexpected error while uploading file: {file.filename}")

        raise HTTPException(
            status_code=500, detail="Internal Server Error during file upload"
        )

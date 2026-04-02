import asyncio
from utils.logger import get_logger

logger = get_logger("app")

background_tasks = set()

async def shutdown_handler(timeout: int = 10):
    logger.info("Shutdown initiated. Waiting for tasks...")

    if not background_tasks:
        logger.info("No background tasks. Exiting.")
        return

    try:
        await asyncio.wait_for(
            asyncio.gather(*background_tasks, return_exceptions=True),
            timeout=timeout
        )
        logger.info("All tasks completed gracefully.")

    except asyncio.TimeoutError:
        logger.warning("Shutdown timeout reached. Cancelling tasks...")

        for task in background_tasks:
            task.cancel()

        await asyncio.gather(*background_tasks, return_exceptions=True)

        logger.info("All tasks cancelled.")
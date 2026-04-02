import asyncio
from utils.logger import get_logger

logger = get_logger("mcp-rag")

background_tasks = set()


def add_task(task: asyncio.Task):
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)


async def shutdown_handler():
    logger.info("MCP RAG shutting down. Waiting for tasks...")

    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)

    logger.info("MCP RAG shutdown complete.")
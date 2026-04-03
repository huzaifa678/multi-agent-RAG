from tools.rag.client import close_rag_client, get_rag_client
from tools.web.client import close_web_client, get_web_client
from utils.logger import get_logger


logger = get_logger(__name__)


app_graph = None

class Runtime:
    def __init__(self):
        pass

    async def init(self):
        logger.info("Initializing runtime...")
        await get_rag_client()
        await get_web_client()

    async def shutdown(self):
        logger.info("Shutting down runtime...")
        await close_web_client()
        await close_rag_client()
        logger.info("Clients closed. Shutdown complete")

runtimeObject = Runtime()
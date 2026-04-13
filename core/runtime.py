from tools.memory.client import get_memory_client
from tools.rag.client import close_rag_client, get_rag_client
from tools.web.client import close_web_client, get_web_client
from utils.logger import get_logger

logger = get_logger(__name__)

app_graph = None


class Runtime:
    def __init__(self):
        self.ready = False
        self.rag_active = False
        self.web_active = False
        self.memory_active = False

    async def init(self):

        logger.info("Initializing runtime...")

        try:
            await get_rag_client()
            self.rag_active = True
            logger.info("RAG client initialized")
        except Exception as e:
            logger.error(f"Failed to start RAG client (MCP server offline?): {e}")

        try:
            await get_web_client()
            self.web_active = True
            logger.info("Web client initialized")
        except Exception as e:
            logger.error(f"Failed to start Web client: {e}")

        try:
            await get_memory_client()
            self.memory_active = True
            logger.info("Memory client initialized")
        except Exception as e:
            logger.error(f"Failed to start Web client: {e}")

        self.ready = True
        logger.info("Runtime initialized and ready")

    async def shutdown(self):
        logger.info("Shutting down runtime...")
        await close_web_client()
        await close_rag_client()
        logger.info("Clients closed. Shutdown complete")


runtimeObject = Runtime()

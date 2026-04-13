from .rag_agent import rag_chain
from .web_agent import web_chain
from .memory_agent import memory_chain
from .aggregator_agent import aggregate_response


class Agents:
    """
    Central registry for all agents.
    Used by LangGraph nodes.
    """

    def __init__(self):

        self.rag = rag_chain

        self.web = web_chain

        self.memory = memory_chain

        self.aggregator = aggregate_response


agents = Agents()

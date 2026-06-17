"""
Shared pytest fixtures and test-time environment setup.

Dummy API keys are injected *before* any project module is imported so that
``utils.config.Config`` (and the LLM clients built from it at import time) can
be constructed without a real ``.env`` or network access.
"""

import os

# Populate credentials the agent modules read at import time. setdefault keeps
# any real value already present in the environment / .env.
for _key in (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "TAVILY_API_KEY",
):
    os.environ.setdefault(_key, f"test-{_key.lower()}")


class FakeChain:
    """Stand-in for a LangChain runnable exposing async ``ainvoke``."""

    def __init__(self, return_value):
        self._return_value = return_value
        self.calls = []

    async def ainvoke(self, payload):
        self.calls.append(payload)
        return self._return_value

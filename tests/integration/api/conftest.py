"""Shared fixtures for API endpoint tests.

Assembles a minimal FastAPI app from the routers, skipping the production
lifespan so the runtime and MCP clients are never started.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.chat as chat_api
import api.health as health_api
import api.upload as upload_api


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(chat_api.router)
    app.include_router(upload_api.router)
    app.include_router(health_api.router)
    return TestClient(app)

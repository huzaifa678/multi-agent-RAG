"""Integration tests for the /chat endpoint."""

import pytest
from fastapi import HTTPException

import api.chat as chat_api

pytestmark = pytest.mark.integration


def test_chat_success(client, monkeypatch):
    monkeypatch.setattr(chat_api.runtimeObject, "ready", True)

    async def fake_handle(payload):
        return {"answer": f"echo:{payload.query}", "session_id": payload.session_id}

    monkeypatch.setattr(chat_api, "handle_chat", fake_handle)

    resp = client.post("/chat", json={"query": "hi", "session_id": "s1"})

    assert resp.status_code == 200
    assert resp.json() == {"answer": "echo:hi", "session_id": "s1"}


def test_chat_returns_503_when_not_ready(client, monkeypatch):
    monkeypatch.setattr(chat_api.runtimeObject, "ready", False)

    resp = client.post("/chat", json={"query": "hi", "session_id": "s1"})

    assert resp.status_code == 503
    assert resp.json()["detail"] == "Service warming up"


def test_chat_validation_error_on_missing_fields(client, monkeypatch):
    monkeypatch.setattr(chat_api.runtimeObject, "ready", True)

    resp = client.post("/chat", json={"query": "hi"})  # missing session_id

    assert resp.status_code == 422


def test_chat_propagates_http_exception(client, monkeypatch):
    monkeypatch.setattr(chat_api.runtimeObject, "ready", True)

    async def boom(payload):
        raise HTTPException(status_code=404, detail="session not found")

    monkeypatch.setattr(chat_api, "handle_chat", boom)

    resp = client.post("/chat", json={"query": "hi", "session_id": "s1"})

    assert resp.status_code == 404
    assert resp.json()["detail"] == "session not found"


def test_chat_unexpected_error_becomes_500(client, monkeypatch):
    monkeypatch.setattr(chat_api.runtimeObject, "ready", True)

    async def boom(payload):
        raise RuntimeError("workflow exploded")

    monkeypatch.setattr(chat_api, "handle_chat", boom)

    resp = client.post("/chat", json={"query": "hi", "session_id": "s1"})

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal Server Error"

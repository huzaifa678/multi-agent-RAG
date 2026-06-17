"""Integration tests for the /upload-doc endpoint."""

import pytest

import api.upload as upload_api

pytestmark = pytest.mark.integration


def test_upload_success(client, monkeypatch):
    async def fake_upload(file, background_tasks):
        return {"file_id": "abc-123", "status": "processing"}

    monkeypatch.setattr(upload_api, "handle_upload", fake_upload)

    resp = client.post(
        "/upload-doc",
        files={"file": ("doc.txt", b"hello world", "text/plain")},
    )

    assert resp.status_code == 200
    assert resp.json() == {"file_id": "abc-123", "status": "processing"}


def test_upload_requires_file(client):
    resp = client.post("/upload-doc")
    assert resp.status_code == 422


def test_upload_unexpected_error_becomes_500(client, monkeypatch):
    async def boom(file, background_tasks):
        raise RuntimeError("disk full")

    monkeypatch.setattr(upload_api, "handle_upload", boom)

    resp = client.post(
        "/upload-doc",
        files={"file": ("doc.txt", b"data", "text/plain")},
    )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal Server Error during file upload"

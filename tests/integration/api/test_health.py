"""Integration tests for the /health endpoint."""

import pytest

pytestmark = pytest.mark.integration


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "timestamp" in body

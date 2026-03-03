from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app, settings


def test_admin_endpoints_require_token_when_configured():
    client = TestClient(app)
    original = settings.admin_api_token
    settings.admin_api_token = "admin-secret"
    try:
        unauthorized = client.post("/api/admin/reindex")
        assert unauthorized.status_code == 401
        assert unauthorized.json()["detail"] == "unauthorized admin token"

        wrong = client.get("/api/admin/metrics", headers={"x-admin-token": "wrong"})
        assert wrong.status_code == 401

        authorized_reindex = client.post("/api/admin/reindex", headers={"x-admin-token": "admin-secret"})
        assert authorized_reindex.status_code == 200

        authorized_metrics = client.get("/api/admin/metrics", headers={"x-admin-token": "admin-secret"})
        assert authorized_metrics.status_code == 200
    finally:
        settings.admin_api_token = original

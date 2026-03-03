from __future__ import annotations

import httpx

from app.config import DOCS_DIR, Settings
from app.services.service_client import ServiceClient
from app.services.store import DocumentStore


def _local_settings() -> Settings:
    return Settings(
        service_call_mode="local",
        use_mock_generation=True,
        docs_dir=DOCS_DIR,
        api_key="",
        api_url="https://example.com/v1/chat/completions",
    )


def test_dependency_health_local_mode():
    settings = _local_settings()
    store = DocumentStore(settings.docs_dir)
    store.load()
    client = ServiceClient(settings=settings, store=store)
    health = client.dependency_health()
    assert health["retrieval-service"]["healthy"] is True
    assert health["generation-service"]["healthy"] is True


def test_service_client_skill_save_and_list():
    settings = _local_settings()
    store = DocumentStore(settings.docs_dir)
    store.load()
    client = ServiceClient(settings=settings, store=store)
    saved = client.save_skill("custom_flow", "步骤A->步骤B->给来源")
    assert saved["name"] == "custom_flow"
    active = client.list_saved_skills()
    assert len(active.skills) >= 1


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)

    def json(self) -> dict:
        return self._payload


class _HttpModeFakeClient:
    def __init__(self, timeout: float):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url: str):
        if "rerank-service" in url:
            raise httpx.ConnectError("rerank timeout")
        return _FakeResponse(200, {"status": "ok"})

    def post(self, url: str, json: dict | None = None):
        if url.endswith("/reindex"):
            return _FakeResponse(200, {"chunks": 88})
        return _FakeResponse(200, {"ok": True})


def _http_settings() -> Settings:
    settings = _local_settings()
    settings.service_call_mode = "http"
    settings.retrieval_service_url = "http://retrieval-service:8001"
    settings.rerank_service_url = "http://rerank-service:8002"
    settings.memory_service_url = "http://memory-service:8003"
    settings.skill_service_url = "http://skill-service:8004"
    settings.generation_service_url = "http://generation-service:8005"
    return settings


def test_dependency_health_http_mode(monkeypatch):
    monkeypatch.setattr("app.services.service_client.httpx.Client", _HttpModeFakeClient)
    settings = _http_settings()
    store = DocumentStore(settings.docs_dir)
    store.load()
    client = ServiceClient(settings=settings, store=store)
    health = client.dependency_health()
    assert health["retrieval-service"]["healthy"] is True
    assert health["rerank-service"]["healthy"] is False
    assert "timeout" in str(health["rerank-service"]["detail"])


def test_reindex_http_mode(monkeypatch):
    monkeypatch.setattr("app.services.service_client.httpx.Client", _HttpModeFakeClient)
    settings = _http_settings()
    store = DocumentStore(settings.docs_dir)
    store.load()
    client = ServiceClient(settings=settings, store=store)
    payload = client.reindex()
    assert payload["chunks"] == 88

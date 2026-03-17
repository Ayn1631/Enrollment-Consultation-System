from __future__ import annotations

import httpx

from app.config import DOCS_DIR, Settings
from app.contracts import MemoryEntry, RagQueryResponse
from app.services.service_client import ServiceClient


def _local_settings() -> Settings:
    return Settings(
        service_call_mode="local",
        use_mock_generation=True,
        docs_dir=DOCS_DIR,
        api_key="mock-key",
        api_url="https://example.com/v1/chat/completions",
    )


def test_dependency_health_local_mode():
    settings = _local_settings()
    client = ServiceClient(settings=settings)
    client.startup()
    health = client.dependency_health()
    assert health["rag-agent-service"]["healthy"] is True
    assert health["generation-service"]["healthy"] is True


def test_service_client_skill_save_and_list():
    settings = _local_settings()
    client = ServiceClient(settings=settings)
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
        if "memory-service" in url:
            raise httpx.ConnectError("memory timeout")
        return _FakeResponse(200, {"status": "ok"})

    def post(self, url: str, json: dict | None = None):
        if url.endswith("/rag/reindex"):
            return _FakeResponse(200, {"status": "ok", "chunks": 88, "updated_at": "2026-03-03T00:00:00"})
        if url.endswith("/rag/query"):
            return _FakeResponse(
                200,
                {
                    "trace_id": "trace-rag",
                    "status": "ok",
                    "context_blocks": ["ctx1"],
                    "sources": [
                        {
                            "chunk_id": "c1",
                            "title": "招生章程",
                            "url": "https://example.com",
                            "text": "",
                            "score": 0.9,
                        }
                    ],
                    "degrade_reason": None,
                    "latency_ms": {},
                },
            )
        return _FakeResponse(200, {"ok": True})


def _http_settings() -> Settings:
    settings = _local_settings()
    settings.service_call_mode = "http"
    settings.rag_agent_service_url = "http://rag-agent-service:8001"
    settings.memory_service_url = "http://memory-service:8003"
    settings.skill_service_url = "http://skill-service:8004"
    settings.generation_service_url = "http://generation-service:8005"
    return settings


def test_dependency_health_http_mode(monkeypatch):
    monkeypatch.setattr("app.services.service_client.httpx.Client", _HttpModeFakeClient)
    settings = _http_settings()
    client = ServiceClient(settings=settings)
    health = client.dependency_health()
    assert health["rag-agent-service"]["healthy"] is True
    assert health["memory-service"]["healthy"] is False
    assert "timeout" in str(health["memory-service"]["detail"])


def test_reindex_http_mode(monkeypatch):
    monkeypatch.setattr("app.services.service_client.httpx.Client", _HttpModeFakeClient)
    settings = _http_settings()
    client = ServiceClient(settings=settings)
    payload = client.reindex()
    assert payload["chunks"] == 88


def test_run_rag_graph_http_mode(monkeypatch):
    monkeypatch.setattr("app.services.service_client.httpx.Client", _HttpModeFakeClient)
    settings = _http_settings()
    client = ServiceClient(settings=settings)
    response = client.run_rag_graph(session_id="s1", query="招生章程", top_k=3, debug=False)
    assert isinstance(response, RagQueryResponse)
    assert response.trace_id == "trace-rag"
    assert len(response.context_blocks) >= 1


def test_plan_features_orders_citation_guard_last():
    settings = _local_settings()
    client = ServiceClient(settings=settings)
    ordered = client.plan_features(["citation_guard", "rag", "web_search"])
    assert ordered[0] == "rag"
    assert ordered[-1] == "citation_guard"


def test_execute_skill_prefers_langchain4j_bridge(monkeypatch):
    settings = _local_settings()
    settings.langchain4j_service_url = "http://langchain4j-service:8080"
    client = ServiceClient(settings=settings)

    class _FakeBridge:
        def execute(self, query, session_id, saved_skill_id):
            return "来自LangChain4j的技能结果"

    monkeypatch.setattr(client, "_langchain4j_bridge", _FakeBridge())
    result = client.execute_skill(query="招生政策", session_id="s1", saved_skill_id="skill-v1")
    assert result.note == "来自LangChain4j的技能结果"


def test_memory_client_supports_long_and_special_memory():
    settings = _local_settings()
    client = ServiceClient(settings=settings)
    client.write_memory(
        session_id="s-memory",
        entry=MemoryEntry(key="response_style", value="偏好简短回答", kind="special", confidence=0.88),
    )
    client.append_long_memory_summary("s-memory", "用户关注学费和资助")
    special_entries = client.read_memory("s-memory", kind="special").entries
    long_entries = client.read_memory("s-memory", kind="long").entries
    assert special_entries[0].value == "偏好简短回答"
    assert "用户关注学费和资助" in long_entries[0].value

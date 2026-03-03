from __future__ import annotations

import httpx

from app.config import DOCS_DIR, Settings
from app.services.service_client import ServiceClient
from app.services.store import ChunkRecord, DocumentStore


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


def test_plan_features_orders_citation_guard_last():
    settings = _local_settings()
    store = DocumentStore(settings.docs_dir)
    store.load()
    client = ServiceClient(settings=settings, store=store)
    ordered = client.plan_features(["citation_guard", "rag", "web_search"])
    assert ordered[0] == "rag"
    assert ordered[-1] == "citation_guard"


def test_execute_skill_prefers_langchain4j_bridge(monkeypatch):
    settings = _local_settings()
    settings.langchain4j_service_url = "http://langchain4j-service:8080"
    store = DocumentStore(settings.docs_dir)
    store.load()
    client = ServiceClient(settings=settings, store=store)

    class _FakeBridge:
        def execute(self, query, session_id, saved_skill_id):
            return "来自LangChain4j的技能结果"

    monkeypatch.setattr(client, "_langchain4j_bridge", _FakeBridge())
    result = client.execute_skill(query="招生政策", session_id="s1", saved_skill_id="skill-v1")
    assert result.note == "来自LangChain4j的技能结果"


def test_retrieve_augments_with_neo4j_facts(monkeypatch):
    settings = _local_settings()
    settings.rag_stack = "langchain"
    settings.neo4j_uri = "bolt://localhost:7687"
    settings.neo4j_user = "neo4j"
    settings.neo4j_password = "password"
    store = DocumentStore(settings.docs_dir)
    store.load()
    client = ServiceClient(settings=settings, store=store)

    sample_chunk = ChunkRecord(
        chunk_id="sample-1",
        title="样例",
        url="https://example.com",
        text="样例文本",
        tokens=["样", "例"],
        term_freq={"样": 1, "例": 1},
        score=1.0,
        bm25_score=1.0,
        vector_score=1.0,
        keyword_score=1.0,
    )
    monkeypatch.setattr(client._rag_adapter, "retrieve", lambda query, top_k: [sample_chunk])

    class _FakeNeo4j:
        def enabled(self):
            return True

        def fetch_facts(self, query, limit=2):
            return ["专业A关联学院B"]

    monkeypatch.setattr(client, "_neo4j_adapter", _FakeNeo4j())

    response = client.retrieve("招生专业", top_k=3)
    titles = [item.title for item in response.chunks]
    assert "Neo4j知识图谱" in titles

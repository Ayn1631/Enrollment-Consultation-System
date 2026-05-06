from __future__ import annotations

import httpx
import pytest
from types import SimpleNamespace

from app.config import Settings
from app.contracts import GenerationRequest, MemoryEntry, RagQueryResponse
from app.services.service_client import ServiceClient
from app.services.llm import GenerationService


def _local_settings(isolated_runtime_settings: Settings) -> Settings:
    return isolated_runtime_settings.model_copy(
        update={
            "service_call_mode": "local",
            "use_mock_generation": True,
        }
    )


def test_dependency_health_local_mode(isolated_runtime_settings: Settings):
    settings = _local_settings(isolated_runtime_settings)
    client = ServiceClient(settings=settings)
    client.startup()
    health = client.dependency_health()
    assert health["rag-agent-service"]["healthy"] is True
    assert health["generation-service"]["healthy"] is True


def test_service_client_skill_save_and_list(isolated_runtime_settings: Settings):
    settings = _local_settings(isolated_runtime_settings)
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

    def post(self, url: str, json: dict | None = None, params: dict | None = None):
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


def _http_settings(runtime_settings: Settings) -> Settings:
    settings = runtime_settings.model_copy(
        update={
            "service_call_mode": "http",
            "use_mock_generation": True,
        }
    )
    settings.rag_agent_service_url = "http://rag-agent-service:8001"
    settings.memory_service_url = "http://memory-service:8003"
    settings.skill_service_url = "http://skill-service:8004"
    settings.generation_service_url = "http://generation-service:8005"
    return settings


def test_dependency_health_http_mode(monkeypatch, runtime_settings: Settings):
    monkeypatch.setattr("app.services.service_client.httpx.Client", _HttpModeFakeClient)
    settings = _http_settings(runtime_settings)
    client = ServiceClient(settings=settings)
    health = client.dependency_health()
    assert health["rag-agent-service"]["healthy"] is True
    assert health["memory-service"]["healthy"] is False
    assert "timeout" in str(health["memory-service"]["detail"])


def test_reindex_http_mode(monkeypatch, runtime_settings: Settings):
    monkeypatch.setattr("app.services.service_client.httpx.Client", _HttpModeFakeClient)
    settings = _http_settings(runtime_settings)
    client = ServiceClient(settings=settings)
    payload = client.reindex()
    assert payload["chunks"] == 88


def test_reindex_http_mode_passes_progress_flag(monkeypatch, runtime_settings: Settings):
    captured_params: list[dict | None] = []

    class _CaptureClient(_HttpModeFakeClient):
        def post(self, url: str, json: dict | None = None, params: dict | None = None):
            captured_params.append(params)
            return super().post(url, json=json, params=params)

    monkeypatch.setattr("app.services.service_client.httpx.Client", _CaptureClient)
    settings = _http_settings(runtime_settings)
    client = ServiceClient(settings=settings)
    payload = client.reindex(show_progress=True)

    assert payload["chunks"] == 88
    assert captured_params[-1] == {"show_progress": "true"}


def test_run_rag_graph_http_mode(monkeypatch, runtime_settings: Settings):
    monkeypatch.setattr("app.services.service_client.httpx.Client", _HttpModeFakeClient)
    settings = _http_settings(runtime_settings)
    client = ServiceClient(settings=settings)
    response = client.run_rag_graph(session_id="s1", query="招生章程", top_k=3, debug=False)
    assert isinstance(response, RagQueryResponse)
    assert response.trace_id == "trace-rag"
    assert len(response.context_blocks) >= 1


def test_query_admissions_structured_returns_major_catalog_result(isolated_runtime_settings: Settings):
    settings = _local_settings(isolated_runtime_settings)
    client = ServiceClient(settings=settings)

    class _FakeRepository:
        def search_major_catalog(self, *, raw_query: str, filters: dict[str, str], limit: int = 8):
            return [
                {
                    "source_file": "2025年招生专业详情.xlsx",
                    "major_name": "自动化",
                    "college_name": "自动化与电气工程学院",
                    "evidence_text": "专业名称：自动化；学费（元）：5500；所在院系：自动化与电气工程学院",
                }
            ]

        def search_score_lines(self, *, raw_query: str, filters: dict[str, str], limit: int = 8):
            return []

        def search_policy_tables(self, *, raw_query: str, filters: dict[str, str], limit: int = 12):
            return []

    client._admissions_toolset.repository = _FakeRepository()
    response = client.query_admissions_structured(query="自动化专业学费是多少？")

    assert response is not None
    assert response.status == "ok"
    assert response.sources[0].title == "自动化 - 自动化与电气工程学院"


def test_plan_features_orders_citation_guard_last(isolated_runtime_settings: Settings):
    settings = _local_settings(isolated_runtime_settings)
    client = ServiceClient(settings=settings)
    ordered = client.plan_features(["citation_guard", "rag", "web_search"])
    assert ordered[0] == "rag"
    assert ordered[-1] == "citation_guard"


def test_execute_skill_prefers_langchain4j_bridge(monkeypatch, isolated_runtime_settings: Settings):
    settings = _local_settings(isolated_runtime_settings)
    settings.langchain4j_service_url = "http://langchain4j-service:8080"
    client = ServiceClient(settings=settings)

    class _FakeBridge:
        def execute(self, query, session_id, saved_skill_id):
            return "来自LangChain4j的技能结果"

    monkeypatch.setattr(client, "_langchain4j_bridge", _FakeBridge())
    result = client.execute_skill(query="招生政策", session_id="s1", saved_skill_id="skill-v1")
    assert result.note == "来自LangChain4j的技能结果"


def test_memory_client_supports_long_and_special_memory(isolated_runtime_settings: Settings):
    settings = _local_settings(isolated_runtime_settings)
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


def test_generate_uses_light_route_and_prompt_cache_in_local_mode(isolated_runtime_settings: Settings):
    settings = _local_settings(isolated_runtime_settings)
    settings.generation_light_model = "light-model"
    settings.generation_main_model = "main-model"
    client = ServiceClient(settings=settings)
    request = GenerationRequest(
        user_query="请介绍招生政策",
        context_blocks=["证据A"],
        feature_notes=["RAG 已执行"],
    )

    first = client.generate(request)
    second = client.generate(request)

    assert first.route == "mock"
    assert first.model == "mock-generator"
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.text == first.text


def test_generate_honors_requested_model_even_in_remote_mode(monkeypatch, runtime_settings: Settings):
    settings = _http_settings(runtime_settings)
    settings.use_mock_generation = False
    if not settings.resolve_llm_api_key():
        pytest.skip("当前环境未配置可用的 LLM API KEY")
    captured_payloads: list[dict] = []

    class _GenerationClient(_HttpModeFakeClient):
        def post(self, url: str, json: dict | None = None):
            if url.endswith("/generate"):
                captured_payloads.append(json or {})
                return _FakeResponse(
                    200,
                    {
                        "text": "生成结果",
                        "model": json.get("model", ""),
                        "route": "requested",
                        "cache_hit": False,
                    },
                )
            return super().post(url, json=json)

    monkeypatch.setattr("app.services.service_client.httpx.Client", _GenerationClient)
    client = ServiceClient(settings=settings)
    response = client.generate(
        GenerationRequest(
            user_query="请详细对比招生政策与报名流程",
            context_blocks=["证据A", "证据B", "证据C"],
            feature_notes=["RAG 已执行"],
            model="custom-model",
        )
    )

    assert response.model == "custom-model"
    assert response.route == "requested"
    assert captured_payloads[0]["model"] == "custom-model"


def test_generation_service_prefers_split_llm_endpoint_and_key(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str, timeout: float, max_retries: int):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["timeout"] = timeout
            captured["max_retries"] = max_retries
            self.api_key = api_key
            self.base_url = base_url
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            captured["create_kwargs"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="这是远程生成结果")
                    )
                ]
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr("app.services.llm.OpenAI", _FakeOpenAI)
    settings = Settings()
    if not settings.resolve_llm_api_key():
        pytest.skip("当前环境未配置可用的 LLM API KEY")
    settings = settings.model_copy(
        update={
            "service_call_mode": "local",
            "use_mock_generation": False,
        }
    )
    service = GenerationService(settings)

    result = service.generate(
        user_query="请介绍招生政策",
        context_blocks=["证据A"],
        feature_notes=["RAG 已执行"],
    )

    assert result.text == "这是远程生成结果"
    assert captured["base_url"] == settings.resolve_llm_api_url()
    assert captured["api_key"] == settings.resolve_llm_api_key()
    assert captured["create_kwargs"]["model"] == settings.generation_light_model

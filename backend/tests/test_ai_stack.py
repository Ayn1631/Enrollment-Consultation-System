from __future__ import annotations

from app.services.ai_stack import LangChain4jSkillBridge, LangGraphFeaturePlanner, Neo4jKnowledgeAdapter


def test_langgraph_planner_fallback_keeps_priority_and_dedup(monkeypatch):
    planner = LangGraphFeaturePlanner()

    def _raise(*args, **kwargs):
        raise RuntimeError("langgraph unavailable")

    monkeypatch.setattr(LangGraphFeaturePlanner, "_plan_with_langgraph", _raise)
    ordered = planner.plan(["citation_guard", "rag", "rag", "skill_exec"])
    assert ordered == ["rag", "skill_exec", "citation_guard"]


class _BridgeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _BridgeHttpClient:
    last_url: str = ""
    last_payload: dict = {}

    def __init__(self, timeout: float):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url: str, json: dict):
        _BridgeHttpClient.last_url = url
        _BridgeHttpClient.last_payload = json
        return _BridgeResponse({"note": "bridge-ok"})


def test_langchain4j_bridge_execute_success(monkeypatch):
    monkeypatch.setattr("app.services.ai_stack.httpx.Client", _BridgeHttpClient)
    bridge = LangChain4jSkillBridge(base_url="http://langchain4j:8080", timeout_seconds=1.2)
    note = bridge.execute(query="招生政策", session_id="s1", saved_skill_id="skill-v1")
    assert note == "bridge-ok"
    assert _BridgeHttpClient.last_url.endswith("/api/skills/execute")
    assert _BridgeHttpClient.last_payload["saved_skill_id"] == "skill-v1"


def test_neo4j_adapter_returns_empty_when_disabled():
    adapter = Neo4jKnowledgeAdapter(uri="", user="", password="", database="neo4j")
    assert adapter.enabled() is False
    assert adapter.fetch_facts("招生") == []

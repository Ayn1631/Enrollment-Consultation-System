from __future__ import annotations

import json
import os

from app.config import Settings
from app.services.ai_stack import LangChain4jSkillBridge, LangGraphFeaturePlanner, Neo4jKnowledgeAdapter
from app.services.ai_stack import load_mcp_server_configs


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


def test_load_mcp_server_configs_supports_cline_style_json(tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "zut-mcp": {
                        "disabled": False,
                        "timeout": 60,
                        "type": "stdio",
                        "command": "uvx",
                        "args": ["zut-mcp@latest"],
                        "env": {"ZUT_FOOD_PATH": r"D:\Mypower\Git\MyPython\MCP\ZUT_food.csv"},
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = Settings(MCP_ENABLED=True, MCP_CONFIG_PATH=str(config_path))

    servers, notes = load_mcp_server_configs(settings)

    assert len(servers) == 1
    assert servers[0].original_name == "zut-mcp"
    assert servers[0].alias == "zut_mcp"
    assert servers[0].transport == "stdio"
    assert servers[0].command == "uvx"
    assert servers[0].args == ["zut-mcp@latest"]
    assert servers[0].env["ZUT_FOOD_PATH"].endswith("ZUT_food.csv")
    assert any("MCP 配置已加载" in note for note in notes)


def test_load_mcp_server_configs_merges_process_env(tmp_path, monkeypatch):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "base_math": {
                        "disabled": False,
                        "timeout": 10,
                        "type": "stdio",
                        "command": "uvx",
                        "args": ["mcp-math"],
                        "env": {"CUSTOM_TOKEN": "abc"},
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    settings = Settings(MCP_ENABLED=True, MCP_CONFIG_PATH=str(config_path))

    servers, _ = load_mcp_server_configs(settings)

    assert len(servers) == 1
    assert "PATH" in servers[0].env
    assert servers[0].env["CUSTOM_TOKEN"] == "abc"

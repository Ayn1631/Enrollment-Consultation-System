from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.models import ChatRequest
from app.services.agent_runtime import AgentRuntime


class _GatewayStub:
    def __init__(self, settings: Settings):
        self.deps = SimpleNamespace(services=SimpleNamespace(settings=settings))

    def _is_time_sensitive_query(self, query: str) -> bool:
        return any(token in query for token in ("最新", "公告", "今天", "现在"))


class _FakeTool:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


def _write_mcp_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_build_plan_should_include_mcp_steps_when_search_capability_matches_query(tmp_path):
    config_path = tmp_path / "mcp.json"
    _write_mcp_config(
        config_path,
        {
            "mcpServers": {
                "bing-search": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "bing-cn-mcp"],
                }
            }
        },
    )
    settings = Settings(MCP_ENABLED=True, MCP_CONFIG_PATH=str(config_path))
    runtime = AgentRuntime(_GatewayStub(settings))
    request = ChatRequest(
        session_id="s1",
        messages=[{"role": "user", "content": "请搜索最新招生公告"}],
        mode="agent",
        features=["rag"],
    )

    steps = runtime.build_plan(
        query="请搜索最新招生公告",
        effective_features=request.features,
        route_label="time_sensitive",
        request=request,
        strategy="quality",
    )

    step_types = [item.step_type for item in steps]
    assert "mcp_discover" in step_types
    assert "mcp_execute" in step_types


def test_build_plan_should_not_include_mcp_steps_when_config_missing(tmp_path):
    config_path = tmp_path / "missing.json"
    settings = Settings(MCP_ENABLED=True, MCP_CONFIG_PATH=str(config_path))
    runtime = AgentRuntime(_GatewayStub(settings))
    request = ChatRequest(
        session_id="s2",
        messages=[{"role": "user", "content": "请搜索最新招生公告"}],
        mode="agent",
        features=["rag"],
    )

    steps = runtime.build_plan(
        query="请搜索最新招生公告",
        effective_features=request.features,
        route_label="time_sensitive",
        request=request,
        strategy="quality",
    )

    step_types = [item.step_type for item in steps]
    assert "mcp_discover" not in step_types
    assert "mcp_execute" not in step_types


def test_select_mcp_tool_should_prefer_search_tool_for_search_query():
    settings = Settings(MCP_ENABLED=False)
    runtime = AgentRuntime(_GatewayStub(settings))
    tools = [
        _FakeTool("fetch_read_page", "read fetched web page"),
        _FakeTool("bing_search_web", "search the web with bing"),
    ]

    selected = runtime._select_mcp_tool(tools, "请搜索最新招生公告")

    assert selected.name == "bing_search_web"

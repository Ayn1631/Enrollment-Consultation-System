from __future__ import annotations

import asyncio
import json
import logging
import os
from types import SimpleNamespace

from app.config import Settings
from app.services.ai_stack import (
    LangChain4jSkillBridge,
    LangGraphFeaturePlanner,
    Neo4jKnowledgeAdapter,
    build_langchain_mcp_runtime,
)
from app.services.ai_stack import _format_exception_summary, load_mcp_server_configs


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


def test_format_exception_summary_should_flatten_exception_group():
    summary = _format_exception_summary(
        ExceptionGroup(
            "group",
            [
                ConnectionError("dns failed"),
                RuntimeError("tool unavailable"),
            ],
        )
    )

    assert "ExceptionGroup" in summary
    assert "ConnectionError: dns failed" in summary
    assert "RuntimeError: tool unavailable" in summary


def test_load_mcp_server_configs_should_log_notes(caplog, tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "broken-http": {
                        "type": "http",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = Settings(MCP_ENABLED=True, MCP_CONFIG_PATH=str(config_path))

    with caplog.at_level(logging.INFO, logger="app.services.ai_stack"):
        servers, notes = load_mcp_server_configs(settings)

    assert servers == []
    assert any("MCP 配置已加载" in note for note in notes)
    assert any("MCP 服务 broken-http 缺少 url/serverUrl，已跳过。" in note for note in notes)
    assert any("[ai_stack.note] MCP 配置已加载" in record.message for record in caplog.records)
    assert any("MCP 服务 broken-http 缺少 url/serverUrl，已跳过。" in record.message for record in caplog.records)


def test_load_mcp_server_configs_should_skip_missing_python_module(tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fetch": {
                        "type": "stdio",
                        "command": "python",
                        "args": ["-m", "definitely_missing_mcp_module_xyz"],
                    },
                    "bing-search": {
                        "type": "stdio",
                        "command": "python",
                        "args": ["-m", "json"],
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = Settings(MCP_ENABLED=True, MCP_CONFIG_PATH=str(config_path))

    servers, notes = load_mcp_server_configs(settings)

    assert len(servers) == 1
    assert servers[0].original_name == "bing-search"
    assert any("缺少 Python 模块 definitely_missing_mcp_module_xyz，已跳过。" in note for note in notes)


def test_build_langchain_mcp_runtime_should_unwrap_bound_tool_name(monkeypatch, tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "bing-search": {
                        "type": "stdio",
                        "command": "python",
                        "args": ["-m", "json"],
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = Settings(MCP_ENABLED=True, MCP_CONFIG_PATH=str(config_path))

    class _WrappedTool:
        def __init__(self):
            self.name = None
            self.description = ""
            self.bound = SimpleNamespace(name="bing_search", description="search the web")

    class _FakeClient:
        def __init__(
            self,
            connections=None,
            *,
            callbacks=None,
            tool_interceptors=None,
            tool_name_prefix=False,
        ):
            self.connections = connections

        async def get_tools(self):
            return [_WrappedTool()]

        async def close(self):
            return None

    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", _FakeClient)

    runtime = asyncio.run(build_langchain_mcp_runtime(settings))

    assert len(runtime.tools) == 1
    assert getattr(runtime.tools[0], "name", "") == "bing_search"
    assert getattr(runtime.tools[0], "description", "") == "search the web"
    asyncio.run(runtime.aclose())


def test_build_langchain_mcp_runtime_should_use_single_multiserver_client(monkeypatch, tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "math-server": {
                        "type": "stdio",
                        "command": "python",
                        "args": ["-m", "json"],
                    },
                    "weather-server": {
                        "type": "http",
                        "url": "https://example.com/mcp",
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = Settings(MCP_ENABLED=True, MCP_CONFIG_PATH=str(config_path))
    captured_connections: list[dict] = []

    class _FakeClient:
        def __init__(
            self,
            connections=None,
            *,
            callbacks=None,
            tool_interceptors=None,
            tool_name_prefix=False,
        ):
            captured_connections.append(connections or {})

        async def get_tools(self):
            return [SimpleNamespace(name="weather_search", description="search weather")]

    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", _FakeClient)

    runtime = asyncio.run(build_langchain_mcp_runtime(settings))

    assert len(captured_connections) == 1
    assert set(captured_connections[0].keys()) == {"math_server", "weather_server"}
    assert len(runtime.tools) == 1
    assert [item.alias for item in runtime.servers] == ["math_server", "weather_server"]


def test_load_mcp_server_configs_should_preserve_streamable_http_transport(tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "web-search-prime": {
                        "type": "streamable_http",
                        "url": "https://example.com/mcp",
                        "headers": {"Authorization": "Bearer test-token"},
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
    assert servers[0].transport == "streamable_http"
    assert servers[0].url == "https://example.com/mcp"
    assert any("MCP 配置已加载" in note for note in notes)


def test_build_langchain_mcp_runtime_should_pass_streamable_http_transport(monkeypatch, tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "web-search-prime": {
                        "type": "streamable_http",
                        "url": "https://example.com/mcp",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = Settings(MCP_ENABLED=True, MCP_CONFIG_PATH=str(config_path))
    captured_connections: list[dict[str, dict[str, str]]] = []

    class _FakeClient:
        def __init__(
            self,
            connections=None,
            *,
            callbacks=None,
            tool_interceptors=None,
            tool_name_prefix=False,
        ):
            captured_connections.append(connections or {})

        async def get_tools(self):
            return [SimpleNamespace(name="web_search_prime", description="search the web")]

        async def close(self):
            return None

    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", _FakeClient)

    runtime = asyncio.run(build_langchain_mcp_runtime(settings))

    assert len(captured_connections) == 1
    assert captured_connections[0]["web_search_prime"]["transport"] == "streamable_http"
    assert captured_connections[0]["web_search_prime"]["url"] == "https://example.com/mcp"
    assert len(runtime.tools) == 1
    asyncio.run(runtime.aclose())

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.models import ChatRequest
from app.services.ai_stack import McpServerConfig
from app.services.agent_runtime import AgentRuntime
import app.services.agent_runtime as agent_runtime_module


class _GatewayStub:
    def __init__(self, settings: Settings):
        self.deps = SimpleNamespace(services=SimpleNamespace(settings=settings))

    def _is_time_sensitive_query(self, query: str) -> bool:
        return any(token in query for token in ("最新", "公告", "今天", "现在"))


class _FakeTool:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


class _FakeMcpRuntime:
    def __init__(self):
        self.tools = [_FakeTool("bing_search_web", "search the web with bing")]
        self.servers = [
            McpServerConfig(
                alias="bing_search",
                original_name="bing-search",
                transport="stdio",
                command="npx",
            )
        ]
        self.notes = ["MCP 外部工具已接入"]
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


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


def test_get_mcp_runtime_should_cache_per_trace(monkeypatch):
    settings = Settings(MCP_ENABLED=True)
    runtime = AgentRuntime(_GatewayStub(settings))
    build_calls: list[_FakeMcpRuntime] = []

    async def _fake_build_langchain_mcp_runtime(_settings):
        fake_runtime = _FakeMcpRuntime()
        build_calls.append(fake_runtime)
        return fake_runtime

    monkeypatch.setattr(agent_runtime_module, "build_langchain_mcp_runtime", _fake_build_langchain_mcp_runtime)

    first = runtime.get_mcp_runtime("trace-1")
    second = runtime.get_mcp_runtime("trace-1")

    assert first is second
    assert len(build_calls) == 1

    runtime.release_mcp_runtime("trace-1")

    assert build_calls[0].close_calls == 1


def test_split_query_should_prefer_llm_result(monkeypatch):
    settings = Settings(MCP_ENABLED=False)
    runtime = AgentRuntime(_GatewayStub(settings))
    captured_messages = {}

    class _FakeResponse:
        content = '["中原工学院学费是多少", "中原工学院住宿费是多少"]'

    class _FakeLlm:
        def invoke(self, messages):
            captured_messages["messages"] = messages
            return _FakeResponse()

    monkeypatch.setattr(agent_runtime_module, "build_langchain_chat_model", lambda *args, **kwargs: _FakeLlm())

    parts = runtime.split_query(
        "中原工学院学费和住宿费分别是多少",
        "quality",
        memory_text="短期记忆：用户上一轮在问本科收费标准。",
    )

    assert parts == ["中原工学院学费是多少", "中原工学院住宿费是多少"]
    assert "短期记忆：用户上一轮在问本科收费标准。" in str(captured_messages["messages"][1].content)


def test_build_plan_should_prefer_llm_result(monkeypatch):
    settings = Settings(MCP_ENABLED=False)
    runtime = AgentRuntime(_GatewayStub(settings))
    request = ChatRequest(
        session_id="s3",
        messages=[{"role": "user", "content": "中原工学院软件工程专业值得报吗"}],
        mode="agent",
        features=["rag"],
    )
    captured_messages = {}

    class _FakeResponse:
        content = (
            '[{"step_type":"local_rag_search","title":"检索专业资料","instruction":"检索软件工程专业的培养、就业、录取和学费信息。"},'
            '{"step_type":"synthesize_step","title":"综合结论","instruction":"基于前面证据判断该专业是否值得报考。"}]'
        )

    class _FakeLlm:
        def invoke(self, messages):
            captured_messages["messages"] = messages
            return _FakeResponse()

    monkeypatch.setattr(agent_runtime_module, "build_langchain_chat_model", lambda *args, **kwargs: _FakeLlm())

    steps = runtime.build_plan(
        query="中原工学院软件工程专业值得报吗",
        effective_features=request.features,
        route_label="policy",
        request=request,
        strategy="quality",
        memory_text="长期记忆：用户明确想报本科理工类专业。",
    )

    assert [item.step_type for item in steps] == ["local_rag_search", "synthesize_step"]
    assert steps[0].instruction == "检索软件工程专业的培养、就业、录取和学费信息。"
    assert "长期记忆：用户明确想报本科理工类专业。" in str(captured_messages["messages"][1].content)

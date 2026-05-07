from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from app.admissions_kb.tools import StructuredToolPayload
from app.config import Settings
from app.models import ChatRequest, ChatSource
from app.services.ai_stack import McpServerConfig, McpToolRuntime
from app.services.agent_types import PlanStep, StepExecutionResult
from app.services.agent_runtime import AgentRuntime
import app.services.agent_runtime as agent_runtime_module


class _GatewayStub:
    def __init__(self, settings: Settings):
        self.deps = SimpleNamespace(
            services=SimpleNamespace(settings=settings),
            container=SimpleNamespace(
                isolation=SimpleNamespace(execute=lambda *_args, **_kwargs: SimpleNamespace(ok=False, value=None, error="unused"))
            ),
        )

    def _is_time_sensitive_query(self, query: str) -> bool:
        return any(token in query for token in ("最新", "公告", "今天", "现在"))

    def _build_langchain_history_messages(self, _messages):
        return []

    def _invoke_rag(self, *_args, **_kwargs):
        return SimpleNamespace(context_blocks=[], sources=[], degrade_reason=None)

    def _dedupe_chat_sources(self, sources: list[ChatSource], limit: int = 5) -> list[ChatSource]:
        return sources[:limit]

    def _audit_user_input(self, query: str):
        return False, "", query

    def _audit_generated_output(self, text: str):
        return False, "", text

    def _route_features(self, _query: str, request: ChatRequest):
        return SimpleNamespace(features=request.features, route_label="policy", reason="test", audit=[], notes=[])

    def _classify_query_intent(self, _query: str):
        return "policy", "test"

    def _guard_skill_request(self, query: str, saved_skill_id: str | None):
        return True, "allowed"

    def _invoke_skill(self, query: str, session_id: str, saved_skill_id: str | None, fail_features: set[str]):
        return "skill-result"


class _FakeTool:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


class _WrappedFakeTool:
    def __init__(self, name: str, description: str = ""):
        self.name = None
        self.description = ""
        self.bound = _FakeTool(name, description)


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


def test_build_plan_should_fallback_to_goal_list_when_llm_unavailable(monkeypatch):
    settings = Settings(MCP_ENABLED=False)
    runtime = AgentRuntime(_GatewayStub(settings))
    monkeypatch.setattr(agent_runtime_module, "build_langchain_chat_model", lambda *args, **kwargs: None)
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

    assert len(steps) == 2
    assert all(item.goal for item in steps)
    assert "请搜索最新招生公告" in steps[0].goal
    assert "不确定性" in steps[-1].goal


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
        content = '["梳理软件工程专业的培养、就业、录取和学费相关证据。","基于前面证据判断该专业是否值得报考，并说明不确定性。"]'

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

    assert [item.goal for item in steps] == [
        "梳理软件工程专业的培养、就业、录取和学费相关证据。",
        "基于前面证据判断该专业是否值得报考，并说明不确定性。",
    ]
    assert "长期记忆：用户明确想报本科理工类专业。" in str(captured_messages["messages"][1].content)


def test_rewrite_query_should_include_recent_user_context_and_restore_missing_constraints(monkeypatch):
    settings = Settings(MCP_ENABLED=False)
    runtime = AgentRuntime(_GatewayStub(settings))
    request = ChatRequest(
        session_id="s-rewrite-1",
        messages=[
            {"role": "user", "content": "我河南高考500分, 可以上什么专业???"},
            {"role": "assistant", "content": "你是文科还是理科？"},
            {"role": "user", "content": "2026, 物理类. 你需要根据25的情况给我一个推荐"},
        ],
        mode="agent",
        features=["rag"],
    )
    captured_messages = {}

    class _FakeResponse:
        content = "如果参考2025年的录取情况，河南物理类考生2026年高考多少分可以推荐报考中原工学院的哪些专业？"

    class _FakeLlm:
        def invoke(self, messages):
            captured_messages["messages"] = messages
            return _FakeResponse()

    monkeypatch.setattr(agent_runtime_module, "build_langchain_chat_model", lambda *args, **kwargs: _FakeLlm())

    rewritten = runtime.rewrite_query(
        request=request,
        last_user="2026, 物理类. 你需要根据25的情况给我一个推荐",
        memory_text="当前没有可用记忆。",
        strategy="quality",
    )

    assert "我河南高考500分, 可以上什么专业???" in str(captured_messages["messages"][1].content)
    assert "500分" in rewritten
    assert "河南" in rewritten
    assert "物理类" in rewritten
    assert "2025年" in rewritten or "25" in rewritten
    assert "2026" in rewritten


def test_build_plan_should_append_synthesis_goal_when_missing(monkeypatch):
    settings = Settings(MCP_ENABLED=False)
    runtime = AgentRuntime(_GatewayStub(settings))
    monkeypatch.setattr(runtime, "_has_mcp_servers", lambda: True)
    request = ChatRequest(
        session_id="s4",
        messages=[{"role": "user", "content": "帮我看 2025 年招生录取情况"}],
        mode="agent",
        features=["rag"],
    )

    class _FakeResponse:
        content = '["先确认校内已有的招生章程和专业资料是否足够。","如果现有依据不足，再补强近年录取或招生公告相关公开信息。"]'

    class _FakeLlm:
        def invoke(self, _messages):
            return _FakeResponse()

    monkeypatch.setattr(agent_runtime_module, "build_langchain_chat_model", lambda *args, **kwargs: _FakeLlm())

    steps = runtime.build_plan(
        query="帮我看 2025 年招生录取情况",
        effective_features=request.features,
        route_label="policy",
        request=request,
        strategy="quality",
    )

    assert len(steps) == 3
    assert "补强近年录取或招生公告相关公开信息" in steps[1].goal
    assert "不确定性" in steps[-1].goal


def test_build_plan_prompt_should_surface_open_fact_and_rag_catalog_guidance(monkeypatch):
    settings = Settings(MCP_ENABLED=False)
    runtime = AgentRuntime(_GatewayStub(settings))
    monkeypatch.setattr(runtime, "_has_mcp_servers", lambda: True)
    monkeypatch.setattr(runtime, "_get_rag_document_catalog_text", lambda: "04-学院与专业概览.md；学校概况.md")
    request = ChatRequest(
        session_id="s5",
        messages=[{"role": "user", "content": "人工智能学院的院长是谁"}],
        mode="agent",
        features=["rag"],
    )
    captured_messages = {}

    class _FakeResponse:
        content = '["先确认本地资料中是否已有答案。","如果现有证据不足，再补充公开事实并核验。","综合证据给出结论。"]'

    class _FakeLlm:
        def invoke(self, messages):
            captured_messages["messages"] = messages
            return _FakeResponse()

    monkeypatch.setattr(agent_runtime_module, "build_langchain_chat_model", lambda *args, **kwargs: _FakeLlm())

    runtime.build_plan(
        query="人工智能学院的院长是谁",
        effective_features=request.features,
        route_label="policy",
        request=request,
        strategy="quality",
    )

    system_prompt = str(captured_messages["messages"][0].content)
    human_prompt = str(captured_messages["messages"][1].content)
    assert "院系领导" in system_prompt
    assert "开放网页事实查询" in system_prompt
    assert "本地知识库文档清单" in human_prompt
    assert "04-学院与专业概览.md" in human_prompt
    assert "step_type" not in system_prompt
    assert "当前可用工具能力" in human_prompt


def test_review_step_should_request_more_evidence_when_synthesis_uncertain_without_mcp(monkeypatch):
    settings = Settings(MCP_ENABLED=False)
    runtime = AgentRuntime(_GatewayStub(settings))
    monkeypatch.setattr(runtime, "_has_mcp_servers", lambda: True)

    review = runtime.review_step(
        PlanStep("综合已有证据并给出结论。"),
        StepExecutionResult(ok=True, message="目前不能据此确定人工智能学院院长是谁。"),
        is_final_step=True,
        accumulated_tool_audit=["agent_tool:local_rag_search"],
    )

    assert review.ok is False
    assert "尚未使用 MCP 外部工具补强" in review.message


def test_run_subproblem_agent_should_offer_all_tools_to_react_agent(monkeypatch):
    settings = Settings(MCP_ENABLED=True)
    runtime = AgentRuntime(_GatewayStub(settings))
    request = ChatRequest(
        session_id="s6",
        messages=[{"role": "user", "content": "人工智能学院院长是谁"}],
        mode="agent",
        features=["rag"],
    )
    captured = {}

    class _FakeLlm:
        pass

    class _FakeAiMessage:
        type = "ai"
        content = "已完成当前步骤。"

    class _FakeAgent:
        async def ainvoke(self, payload):
            captured["payload"] = payload
            return {"messages": [_FakeAiMessage()]}

    def _fake_create_react_agent(_llm, tools, prompt=None, version=None):
        captured["tool_names"] = [getattr(item, "name", "") for item in tools]
        captured["prompt"] = prompt
        captured["version"] = version
        return _FakeAgent()

    monkeypatch.setattr(agent_runtime_module, "build_langchain_chat_model", lambda *args, **kwargs: _FakeLlm())
    monkeypatch.setattr(runtime, "get_mcp_runtime", lambda _trace_id: McpToolRuntime(
        client=None,
        tools=[_WrappedFakeTool("bing_search", "search official pages"), _FakeTool("weather_lookup", "weather tool")],
        servers=[McpServerConfig(alias="bing_search", original_name="bing-search", transport="stdio", command="npx")],
        notes=[],
    ))
    monkeypatch.setattr("langgraph.prebuilt.create_react_agent", _fake_create_react_agent)

    result = runtime.run_subproblem_agent(
        step=PlanStep("先补齐回答当前问题所需的证据。"),
        plan_step_index=1,
        total_plan_steps=2,
        subproblem=SimpleNamespace(
            query="人工智能学院院长是谁",
            context_blocks=[],
            sources=[],
            notes=[],
            tool_audit=[],
        ),
        request=request,
        fail_features=set(),
        effective_features=["rag"],
        memory_context_blocks=[],
        memory_text="当前没有可用记忆。",
        trace_id="trace-1",
        route_label="policy",
        step_events=[],
        sink=None,
        attempt=1,
    )

    assert result.ok is True
    assert captured["version"] == "v2"
    assert "local_rag_search" in captured["tool_names"]
    assert "major_catalog_lookup" in captured["tool_names"]
    assert "scoreline_lookup" in captured["tool_names"]
    assert "policy_table_lookup" in captured["tool_names"]
    assert "bing_search" in captured["tool_names"]
    assert "weather_lookup" in captured["tool_names"]


def test_build_local_agent_tools_should_include_executable_structured_lookup(monkeypatch):
    settings = Settings(MCP_ENABLED=False)
    runtime = AgentRuntime(_GatewayStub(settings))
    request = ChatRequest(
        session_id="s6-local",
        messages=[{"role": "user", "content": "自动化专业学费多少"}],
        mode="agent",
        features=["rag"],
    )
    collector_context_blocks: list[str] = []
    collector_sources: list[ChatSource] = []
    collector_notes: list[str] = []
    collector_tool_audit: list[str] = []

    def _tool_factory(name: str):
        def _decorate(fn):
            fn.name = name
            return fn

        return _decorate

    monkeypatch.setattr(
        agent_runtime_module.StructuredAdmissionsToolset,
        "major_catalog_fulltext",
        lambda self: StructuredToolPayload(
            tool_name="major_catalog_lookup",
            matched_fields=[],
            route_reason="专业目录类结构化全量文本返回",
            records=[
                {
                    "source_file": "2025年招生专业详情.xlsx",
                    "major_code": "080801",
                    "major_name": "自动化",
                    "duration": "四年",
                    "tuition": "5500",
                    "exam_subjects": "物理+化学",
                    "degree_type": "工学",
                    "college_name": "自动化与电气工程学院",
                    "evidence_text": "专业名称：自动化；学费（元）：5500；所在院系：自动化与电气工程学院",
                }
            ],
        ),
    )

    tools = runtime._build_local_agent_tools(
        tool_factory=_tool_factory,
        request=request,
        effective_features=["rag"],
        fail_features=set(),
        memory_context_blocks=[],
        collector_context_blocks=collector_context_blocks,
        collector_sources=collector_sources,
        collector_notes=collector_notes,
        collector_tool_audit=collector_tool_audit,
        rag_document_catalog="招生章程.md",
    )

    major_lookup = next(item for item in tools if getattr(item, "name", "") == "major_catalog_lookup")
    result_text = major_lookup("自动化专业学费多少")

    assert "结构化工具：major_catalog_lookup" in result_text
    assert "以下为 xlsx 原表格式输出" in result_text
    assert "专业代码\t专业名称\t学制\t学费（元）\t选考科目\t学位授予门类\t所在院系" in result_text
    assert any("structured:major_catalog_lookup" in item for item in collector_context_blocks)
    assert any(source.title == "自动化 - 自动化与电气工程学院" for source in collector_sources)
    assert "agent_tool:major_catalog_lookup" in collector_tool_audit


def test_run_subproblem_agent_should_emit_tool_input_and_output_trace(monkeypatch, tmp_path: Path):
    settings = Settings(MCP_ENABLED=False)
    runtime = AgentRuntime(_GatewayStub(settings))
    request = ChatRequest(
        session_id="s-tool-trace",
        messages=[{"role": "user", "content": "自动化专业学费多少"}],
        mode="agent",
        features=["rag"],
    )
    step_events = []
    trace_log_path = tmp_path / "agent_tool_traces.jsonl"

    class _FakePlanningAiMessage:
        type = "ai"
        content = ""
        tool_calls = [
            {
                "id": "call-1",
                "name": "major_catalog_lookup",
                "args": {"tool_query": "自动化专业学费多少"},
            }
        ]

    class _FakeToolMessage:
        type = "tool"
        name = "major_catalog_lookup"
        tool_call_id = "call-1"
        content = "结构化工具：major_catalog_lookup\n命中记录数：1"

    class _FakeFinalAiMessage:
        type = "ai"
        content = "自动化专业学费为 5500 元。"

    class _FakeLlm:
        pass

    class _FakeAgent:
        async def ainvoke(self, payload):
            return {"messages": [_FakePlanningAiMessage(), _FakeToolMessage(), _FakeFinalAiMessage()]}

    monkeypatch.setattr(agent_runtime_module, "build_langchain_chat_model", lambda *args, **kwargs: _FakeLlm())
    monkeypatch.setattr(runtime, "get_mcp_runtime", lambda _trace_id: McpToolRuntime(client=None, tools=[], servers=[], notes=[]))
    monkeypatch.setattr(runtime, "_get_tool_trace_log_path", lambda: trace_log_path)
    monkeypatch.setattr("langgraph.prebuilt.create_react_agent", lambda *_args, **_kwargs: _FakeAgent())

    result = runtime.run_subproblem_agent(
        step=PlanStep("先补齐回答当前问题所需的证据。"),
        plan_step_index=1,
        total_plan_steps=1,
        subproblem=SimpleNamespace(
            subproblem_id="sp-1",
            query="自动化专业学费多少",
            context_blocks=[],
            sources=[],
            notes=[],
            tool_audit=[],
        ),
        request=request,
        fail_features=set(),
        effective_features=["rag"],
        memory_context_blocks=[],
        memory_text="当前没有可用记忆。",
        trace_id="trace-tool-1",
        route_label="policy",
        step_events=step_events,
        sink=None,
        attempt=1,
    )

    assert result.ok is True
    tool_event = next(item for item in step_events if item.title == "工具调用：major_catalog_lookup")
    assert "传入：" in tool_event.message
    assert "自动化专业学费多少" in tool_event.message
    assert "传出：" in tool_event.message
    assert "命中记录数：1" in tool_event.message
    lines = trace_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["trace_id"] == "trace-tool-1"
    assert payload["session_id"] == "s-tool-trace"
    assert payload["tool_name"] == "major_catalog_lookup"
    assert payload["tool_args"]["tool_query"] == "自动化专业学费多少"
    assert "命中记录数：1" in payload["tool_output"]


def test_normalize_agent_tool_should_unwrap_bound_tool_name_and_description():
    runtime = AgentRuntime(_GatewayStub(Settings(MCP_ENABLED=False)))
    wrapped = _WrappedFakeTool("bing_search", "search official pages")

    normalized = runtime._normalize_agent_tool(wrapped)

    assert normalized is wrapped.bound
    assert runtime._get_agent_tool_name(wrapped) == "bing_search"
    assert runtime._get_agent_tool_description(wrapped) == "search official pages"


def test_build_crawl_webpage_payload_should_fill_urlmap_from_cache():
    runtime = AgentRuntime(_GatewayStub(Settings(MCP_ENABLED=False)))

    payload = runtime._build_crawl_webpage_payload(
        uuids=["id_1", "id_2"],
        url_map={"id_1": "https://www.zut.edu.cn/a.htm"},
        cached_url_map={"id_2": "https://www.zut.edu.cn/b.htm"},
    )

    assert payload == {
        "uuids": ["id_1", "id_2"],
        "urlMap": {
            "id_1": "https://www.zut.edu.cn/a.htm",
            "id_2": "https://www.zut.edu.cn/b.htm",
        },
    }


def test_extract_uuid_url_map_should_parse_json_text_blocks():
    runtime = AgentRuntime(_GatewayStub(Settings(MCP_ENABLED=False)))

    result = [
        {
            "type": "text",
            "text": '[{"uuid":"id_1","title":"人工智能学院","url":"https://www.zut.edu.cn/rgznxy/index.htm"}]',
        }
    ]

    assert runtime._extract_uuid_url_map(result) == {
        "id_1": "https://www.zut.edu.cn/rgznxy/index.htm"
    }

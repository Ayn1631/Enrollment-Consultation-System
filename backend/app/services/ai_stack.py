from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import httpx

from app.config import Settings
from app.models import ChatSource, FeatureFlag

logger = logging.getLogger(__name__)


# 关键变量：定义 Agent 功能执行优先级，保证引用校验在 RAG 后执行。
FEATURE_PRIORITY: dict[FeatureFlag, int] = {
    "rag": 1,
    "web_search": 2,
    "skill_exec": 3,
    "use_saved_skill": 4,
    "citation_guard": 5,
}


def _dedupe_features(features: list[FeatureFlag]) -> list[FeatureFlag]:
    """去重并保序，避免 Agent 重复执行同一功能。"""
    return list(dict.fromkeys(features))


class _PlanState(TypedDict):
    remaining: list[FeatureFlag]
    ordered: list[FeatureFlag]


class LangGraphFeaturePlanner:
    """使用 LangGraph 规划功能执行顺序，缺依赖时回退到本地排序。"""

    def plan(self, features: list[FeatureFlag]) -> list[FeatureFlag]:
        # 关键变量：normalized 保存去重后的输入，作为图执行初始状态。
        normalized = _dedupe_features(features)
        if not normalized:
            return []
        try:
            return self._plan_with_langgraph(normalized)
        except Exception:
            return self.fallback_plan(normalized)

    def _plan_with_langgraph(self, features: list[FeatureFlag]) -> list[FeatureFlag]:
        """通过 LangGraph 的 StateGraph 做可解释的执行计划。"""
        from langgraph.graph import END, StateGraph

        def arrange(state: _PlanState) -> _PlanState:
            remaining = list(state["remaining"])
            ordered = list(state["ordered"])
            if not remaining:
                return {"remaining": remaining, "ordered": ordered}

            # 关键变量：next_idx 按优先级和当前顺序联合排序，保证结果稳定。
            next_idx = min(
                range(len(remaining)),
                key=lambda idx: (FEATURE_PRIORITY.get(remaining[idx], 99), idx),
            )
            ordered.append(remaining.pop(next_idx))
            return {"remaining": remaining, "ordered": ordered}

        graph = StateGraph(_PlanState)
        graph.add_node("arrange", arrange)
        graph.set_entry_point("arrange")
        graph.add_conditional_edges("arrange", lambda s: END if not s["remaining"] else "arrange")
        compiled = graph.compile()
        result = compiled.invoke({"remaining": features, "ordered": []})
        return list(result["ordered"])

    def fallback_plan(self, features: list[FeatureFlag]) -> list[FeatureFlag]:
        """LangGraph 不可用时，按同一优先级策略本地降级。"""
        ordered_pairs = [(idx, item) for idx, item in enumerate(features)]
        ordered_pairs.sort(key=lambda pair: (FEATURE_PRIORITY.get(pair[1], 99), pair[0]))
        return [item for _, item in ordered_pairs]


@dataclass(slots=True)
class Neo4jKnowledgeAdapter:
    """从 Neo4j 查询与问题相关的知识图谱事实。"""

    uri: str
    user: str
    password: str
    database: str

    def enabled(self) -> bool:
        """只有 URI 和凭据齐全才尝试查询 Neo4j。"""
        return bool(self.uri and self.user and self.password)

    def fetch_facts(self, query: str, limit: int = 2) -> list[str]:
        """按查询词拉取图谱事实，异常时返回空列表并由网关降级。"""
        if not self.enabled():
            return []
        try:
            from neo4j import GraphDatabase
        except Exception:
            return []

        cypher = """
        CALL db.index.fulltext.queryNodes('admission_index', $query) YIELD node, score
        RETURN coalesce(node.name, node.title, '未知节点') AS name,
               coalesce(node.summary, node.text, '') AS summary,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        facts: list[str] = []
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with driver.session(database=self.database) as session:
                result = session.run(cypher, query=query, limit=limit)
                for row in result:
                    # 关键变量：fact_text 是写入上下文的统一字符串格式。
                    fact_text = f"{row.get('name', '')}: {row.get('summary', '')}".strip(": ").strip()
                    if fact_text:
                        facts.append(fact_text)
            driver.close()
        except Exception:
            return []
        return facts


@dataclass(slots=True)
class LangChain4jSkillBridge:
    """通过 HTTP 调用外部 LangChain4j 服务执行历史技能。"""

    base_url: str
    timeout_seconds: float

    def execute(self, query: str, session_id: str, saved_skill_id: str) -> str | None:
        """调用 LangChain4j 的技能端点，成功时返回说明文本。"""
        if not self.base_url or not saved_skill_id:
            return None

        endpoint = f"{self.base_url.rstrip('/')}/api/skills/execute"
        payload = {
            "query": query,
            "session_id": session_id,
            "saved_skill_id": saved_skill_id,
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                body = response.json()
        except Exception:
            return None

        note = body.get("note") or body.get("answer") or body.get("result")
        if not note:
            return None
        return str(note)


@dataclass(slots=True)
class AgentExecutionResult:
    text: str
    sources: list[ChatSource]
    notes: list[str]
    tool_audit: list[str]


@dataclass(slots=True)
class McpServerConfig:
    alias: str
    original_name: str
    transport: str
    timeout_seconds: float | None = None
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class McpToolRuntime:
    client: Any | None
    tools: list[Any]
    servers: list[McpServerConfig]
    notes: list[str]

    async def aclose(self) -> None:
        if self.client is None:
            return
        close_method = getattr(self.client, "close", None)
        if close_method is None:
            return
        result = close_method()
        if hasattr(result, "__await__"):
            await result


def _expand_text_value(value: str) -> str:
    return os.path.expandvars(value).strip()


def _safe_server_alias(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    normalized = normalized.strip("_")
    return normalized or "mcp_server"


def _format_exception_summary(exc: BaseException) -> str:
    """把异常压平成可读摘要，尤其处理 ExceptionGroup 这种大便输出。"""
    nested = getattr(exc, "exceptions", None)
    if nested:
        child_parts = [
            _format_exception_summary(item)
            for item in nested
            if isinstance(item, BaseException)
        ]
        compact = " | ".join(part for part in child_parts if part)
        return f"{exc.__class__.__name__}: {compact or str(exc)}"
    return f"{exc.__class__.__name__}: {exc}"


def _note_level(message: str) -> int:
    normalized = message.lower()
    if any(token in normalized for token in ("失败", "无效", "不可用", "不存在", "缺少", "已跳过", "关闭")):
        return logging.WARNING
    return logging.INFO


def _append_note(notes: list[str], message: str) -> None:
    notes.append(message)
    logger.log(_note_level(message), "[ai_stack.note] %s", message)


def _single_note(message: str) -> list[str]:
    logger.log(_note_level(message), "[ai_stack.note] %s", message)
    return [message]


def _ensure_mcp_adapter_compatibility(notes: list[str]) -> None:
    """兼容旧版 mcp SDK，补齐 langchain_mcp_adapters 导入时缺失的类型别名。"""
    try:
        import mcp.client.session as session_module
        import mcp.types as types_module
    except Exception as exc:  # noqa: BLE001
        _append_note(notes, f"MCP 兼容补丁加载失败：{exc.__class__.__name__}")
        return

    patched: list[str] = []
    if not hasattr(session_module, "ElicitationFnT") and hasattr(session_module, "SamplingFnT"):
        setattr(session_module, "ElicitationFnT", getattr(session_module, "SamplingFnT"))
        patched.append("session.ElicitationFnT->SamplingFnT")
    if not hasattr(types_module, "ElicitRequestParams") and hasattr(types_module, "CreateMessageRequestParams"):
        setattr(types_module, "ElicitRequestParams", getattr(types_module, "CreateMessageRequestParams"))
        patched.append("types.ElicitRequestParams->CreateMessageRequestParams")
    if not hasattr(types_module, "ElicitResult") and hasattr(types_module, "CreateMessageResult"):
        setattr(types_module, "ElicitResult", getattr(types_module, "CreateMessageResult"))
        patched.append("types.ElicitResult->CreateMessageResult")
    if patched:
        _append_note(notes, "已应用 MCP 兼容补丁：" + ", ".join(patched))


def _normalize_stdio_command(command: str) -> tuple[str, str | None]:
    """对 Windows 上常见 MCP 命令做兼容处理，减少子进程启动失败。"""
    normalized = command.strip()
    if not normalized or os.name != "nt":
        return normalized, None

    lower_name = Path(normalized).name.lower()
    if lower_name in {"python", "python.exe"}:
        return sys.executable, f"已将 MCP 命令 {command} 映射为当前解释器：{sys.executable}"

    if lower_name in {"npx", "npx.cmd", "npm", "npm.cmd"}:
        preferred = "npx.cmd" if lower_name.startswith("npx") else "npm.cmd"
        resolved = shutil.which(preferred) or shutil.which(lower_name)
        if resolved:
            return resolved, f"已将 MCP 命令 {command} 映射为 Windows 可执行文件：{resolved}"

    return normalized, None


def _build_stdio_env(
    *,
    command: str,
    extra_env: dict[str, str],
    config_dir: Path,
) -> tuple[dict[str, str], str | None]:
    """合并 stdio 子进程环境，并为 Windows/npm 注入可写缓存目录。"""
    merged_env = {str(key): str(value) for key, value in os.environ.items()}
    merged_env.update(extra_env)

    if os.name != "nt":
        return merged_env, None

    lower_name = Path(command).name.lower()
    if lower_name not in {"npx", "npx.cmd", "npm", "npm.cmd"}:
        return merged_env, None

    cache_key = next((key for key in ("npm_config_cache", "NPM_CONFIG_CACHE") if merged_env.get(key)), None)
    if cache_key is not None:
        return merged_env, None

    cache_dir = config_dir / ".cache" / "npm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    merged_env["npm_config_cache"] = str(cache_dir)
    return merged_env, f"已为 {Path(command).name} 注入可写 npm 缓存目录：{cache_dir}"


def load_mcp_server_configs(settings: Settings) -> tuple[list[McpServerConfig], list[str]]:
    """读取类似 Cline 的 mcpServers 配置，并归一化为官方 SDK 可消费格式。"""
    notes: list[str] = []
    if not settings.mcp_enabled:
        return [], _single_note("MCP 已被 MCP_ENABLED=false 关闭。")

    config_path = settings.resolve_mcp_config_path()
    if config_path is None:
        return [], _single_note("未找到 MCP 配置文件，已跳过外部 MCP 工具接入。")
    if not config_path.exists():
        return [], _single_note(f"MCP 配置文件不存在：{config_path}")

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [], _single_note(f"MCP 配置文件读取失败：{config_path} ({exc.__class__.__name__})")

    raw_servers = payload.get("mcpServers") if isinstance(payload, dict) else None
    if not isinstance(raw_servers, dict):
        return [], _single_note(f"MCP 配置格式无效：{config_path} 缺少 mcpServers 对象。")

    _append_note(notes, f"MCP 配置已加载：{config_path}")
    servers: list[McpServerConfig] = []
    alias_counts: dict[str, int] = {}
    for original_name, raw in raw_servers.items():
        if not isinstance(raw, dict):
            _append_note(notes, f"MCP 服务 {original_name} 配置不是对象，已跳过。")
            continue
        if bool(raw.get("disabled", False)):
            _append_note(notes, f"MCP 服务 {original_name} 已禁用。")
            continue

        alias_base = _safe_server_alias(str(original_name))
        alias_index = alias_counts.get(alias_base, 0)
        alias_counts[alias_base] = alias_index + 1
        alias = alias_base if alias_index == 0 else f"{alias_base}_{alias_index + 1}"

        transport = str(raw.get("transport") or raw.get("type") or "stdio").strip().lower()
        timeout_seconds = raw.get("timeout")
        normalized_timeout = float(timeout_seconds) if timeout_seconds is not None else None

        if transport == "stdio":
            raw_command = _expand_text_value(str(raw.get("command") or ""))
            command, command_note = _normalize_stdio_command(raw_command)
            if not command:
                _append_note(notes, f"MCP 服务 {original_name} 缺少 command，已跳过。")
                continue
            if command_note:
                _append_note(notes, f"MCP 服务 {original_name}：{command_note}")
            args = [_expand_text_value(str(item)) for item in list(raw.get("args") or [])]
            if (
                Path(command).name.lower().startswith("python")
                and len(args) >= 2
                and args[0] == "-m"
                and not importlib.util.find_spec(args[1])
            ):
                _append_note(notes, f"MCP 服务 {original_name} 缺少 Python 模块 {args[1]}，已跳过。")
                continue
            extra_env = {
                str(key): _expand_text_value(str(value))
                for key, value in dict(raw.get("env") or {}).items()
            }
            merged_env, env_note = _build_stdio_env(
                command=command,
                extra_env=extra_env,
                config_dir=config_path.parent,
            )
            if env_note:
                _append_note(notes, f"MCP 服务 {original_name}：{env_note}")
            servers.append(
                McpServerConfig(
                    alias=alias,
                    original_name=str(original_name),
                    transport="stdio",
                    timeout_seconds=normalized_timeout,
                    command=command,
                    args=args,
                    env=merged_env,
                )
            )
            continue

        if transport in {"http", "sse", "streamable_http"}:
            url = _expand_text_value(str(raw.get("url") or raw.get("serverUrl") or raw.get("endpoint") or ""))
            if not url:
                _append_note(notes, f"MCP 服务 {original_name} 缺少 url/serverUrl，已跳过。")
                continue
            headers = {
                str(key): _expand_text_value(str(value))
                for key, value in dict(raw.get("headers") or {}).items()
            }
            servers.append(
                McpServerConfig(
                    alias=alias,
                    original_name=str(original_name),
                    transport=transport,
                    timeout_seconds=normalized_timeout,
                    url=url,
                    headers=headers,
                )
            )
            continue

        _append_note(notes, f"MCP 服务 {original_name} 使用了未支持的 transport={transport}，已跳过。")
    return servers, notes


async def build_langchain_mcp_runtime(settings: Settings) -> McpToolRuntime:
    """使用官方 MCP + LangChain 适配器构造可直接注入 Agent 的外部工具集。"""
    servers, notes = load_mcp_server_configs(settings)
    if not servers:
        return McpToolRuntime(client=None, tools=[], servers=[], notes=notes)

    _ensure_mcp_adapter_compatibility(notes)

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except Exception as exc:  # noqa: BLE001
        _append_note(notes, f"langchain-mcp-adapters 不可用：{exc.__class__.__name__}")
        return McpToolRuntime(client=None, tools=[], servers=servers, notes=notes)

    connections: dict[str, dict[str, Any]] = {}
    for server in servers:
        if server.transport == "stdio":
            connections[server.alias] = {
                "transport": "stdio",
                "command": server.command,
                "args": server.args,
                "env": server.env,
            }
        else:
            connection: dict[str, Any] = {
                "transport": server.transport,
                "url": server.url,
            }
            if server.headers:
                connection["headers"] = server.headers
            connections[server.alias] = connection

    client = None
    configured_tools: list[Any] = []
    try:
        client = MultiServerMCPClient(connections)
        tools = await client.get_tools()
    except Exception as exc:  # noqa: BLE001
        _append_note(notes, f"MCP 工具加载失败：{_format_exception_summary(exc)}")
        return McpToolRuntime(client=client, tools=[], servers=servers, notes=notes)

    for tool in tools:
        bound_tool = getattr(tool, "bound", None)
        if getattr(tool, "name", None) is None and getattr(bound_tool, "name", None):
            tool = bound_tool
        configured_tools.append(tool)

    if not configured_tools:
        _append_note(notes, "MCP 已成功连接，但未发现可用工具。")
        return McpToolRuntime(client=client, tools=[], servers=servers, notes=notes)

    _append_note(
        notes,
        "MCP 外部工具已接入："
        + ", ".join(f"{item.original_name}->{item.alias}" for item in servers),
    )
    _append_note(notes, f"MCP 共加载 {len(configured_tools)} 个工具。")
    return McpToolRuntime(client=client, tools=configured_tools, servers=servers, notes=notes)


def build_langchain_chat_model(
    settings: Settings,
    *,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
):
    """构造 LangChain ChatOpenAI 兼容模型，缺少依赖或密钥时返回 None。"""
    llm_api_key = settings.resolve_llm_api_key()
    if settings.use_mock_generation or not llm_api_key:
        return None
    base_url = settings.resolve_llm_api_url()
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    model_kwargs: dict[str, float] = {}
    if top_p is not None:
        model_kwargs["top_p"] = top_p
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or settings.generation_main_model,
            api_key=llm_api_key,
            base_url=base_url,
            temperature=0.2 if temperature is None else temperature,
            timeout=settings.llm_timeout_seconds,
            model_kwargs=model_kwargs,
        )
    except Exception:
        pass
    try:
        from langchain_community.chat_models import ChatOpenAI as CommunityChatOpenAI

        logger.warning("langchain_openai 不可用，已回退到 langchain_community.ChatOpenAI。")
        return CommunityChatOpenAI(
            model_name=model or settings.generation_main_model,
            openai_api_key=llm_api_key,
            openai_api_base=base_url,
            temperature=0.2 if temperature is None else temperature,
            request_timeout=settings.llm_timeout_seconds,
            model_kwargs=model_kwargs,
        )
    except Exception:
        return None

from __future__ import annotations

import asyncio
import logging
import re
import threading
import uuid
from datetime import datetime
from time import perf_counter
from typing import Any, Callable

from app.models import AgentStepEvent, AgentStrategy, ChatRequest, ChatSource, FeatureFlag, SessionResult
from app.services.ai_stack import (
    McpToolRuntime,
    build_langchain_chat_model,
    build_langchain_mcp_runtime,
    load_mcp_server_configs,
)
from app.services.agent_types import (
    PlanStep,
    PlanStepType,
    StepExecutionResult,
    StepReviewResult,
    SubproblemState,
)


StepSink = Callable[[AgentStepEvent], None]


class AgentRuntime:
    def __init__(self, gateway: Any):
        self.gateway = gateway
        self.logger = logging.getLogger(__name__)
        self._mcp_runtime_cache: dict[str, McpToolRuntime] = {}
        self._mcp_runtime_locks: dict[str, threading.RLock] = {}
        self._mcp_runtime_guard = threading.RLock()

    @property
    def deps(self):
        return self.gateway.deps

    def new_trace_id(self) -> str:
        return uuid.uuid4().hex

    def get_last_user(self, request: ChatRequest) -> str:
        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "").strip()
        return last_user or "请介绍中原工学院招生政策要点。"

    def emit_step(
        self,
        *,
        events: list[AgentStepEvent],
        sink: StepSink | None,
        strategy: AgentStrategy,
        node: str,
        title: str,
        status: str,
        message: str | None = None,
        subproblem_id: str | None = None,
        plan_step_index: int | None = None,
        attempt: int | None = None,
    ) -> AgentStepEvent:
        event = AgentStepEvent(
            id=uuid.uuid4().hex,
            node=node,
            title=title,
            status=status,  # type: ignore[arg-type]
            message=message,
            subproblem_id=subproblem_id,
            plan_step_index=plan_step_index,
            attempt=attempt,
            strategy=strategy,
            timestamp=datetime.utcnow().isoformat(),
        )
        events.append(event)
        if sink is not None:
            sink(event)
        return event

    def audit_user_input(self, query: str) -> tuple[bool, str, str]:
        return self.gateway._audit_user_input(query)

    def audit_generated_output(self, text: str) -> tuple[bool, str, str]:
        return self.gateway._audit_generated_output(text)

    def route_features(self, query: str, request: ChatRequest):
        return self.gateway._route_features(query, request)

    def dedupe_sources(self, sources: list[ChatSource], limit: int = 5) -> list[ChatSource]:
        return self.gateway._dedupe_chat_sources(sources, limit=limit)

    def build_failure_session(
        self,
        *,
        request: ChatRequest,
        exc: Exception,
        step_events: list[AgentStepEvent] | None = None,
    ) -> SessionResult:
        session = self.gateway._build_agent_failure_session(request=request, exc=exc)
        session.agent_strategy = request.agent_strategy
        session.agent_trace = list(step_events or [])
        return session

    def load_memory_context(self, session_id: str) -> tuple[list[str], str, list[str]]:
        context_blocks: list[str] = []
        notes: list[str] = []
        memory_lines: list[str] = []
        for kind, label, prefix in (
            ("short", "短期记忆", "[memory]"),
            ("long", "长期记忆", "[long-memory]"),
            ("special", "特殊记忆", "[special-memory]"),
        ):
            result = self.deps.container.isolation.execute(
                "memory-service",
                lambda kind=kind: self.deps.services.read_memory(session_id=session_id, kind=kind),
            )
            if not result.ok or result.value is None:
                notes.append(f"{label}读取失败，已忽略。")
                continue
            entries = result.value.entries[:3]
            if not entries:
                continue
            context_blocks.extend([f"{prefix} {item.value}" for item in entries])
            memory_lines.append(f"{label}：")
            memory_lines.extend([f"- {item.key}: {item.value}" for item in entries])
            notes.append(f"{label}已接入上下文。")
        return context_blocks, "\n".join(memory_lines) if memory_lines else "当前没有可用记忆。", notes

    def rewrite_query(
        self,
        *,
        request: ChatRequest,
        last_user: str,
        memory_text: str,
        strategy: AgentStrategy,
    ) -> str:
        llm = build_langchain_chat_model(
            self.deps.services.settings,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        if llm is None:
            return self._rule_rewrite_query(last_user=last_user, memory_text=memory_text, strategy=strategy)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except Exception:
            return self._rule_rewrite_query(last_user=last_user, memory_text=memory_text, strategy=strategy)
        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "你负责把用户问题改写成适合招生咨询工具链执行的单轮查询。"
                            "只输出改写后的问题文本，不要解释。"
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"执行策略：{strategy}\n"
                            f"用户原问题：{last_user}\n\n"
                            f"相关记忆：\n{memory_text}\n\n"
                            "请补全省略代词、保持原意，不要编造新需求。"
                        )
                    ),
                ]
            )
            content = str(getattr(response, "content", "") or "").strip()
            return content or self._rule_rewrite_query(last_user=last_user, memory_text=memory_text, strategy=strategy)
        except Exception:
            return self._rule_rewrite_query(last_user=last_user, memory_text=memory_text, strategy=strategy)

    def split_query(self, query: str, strategy: AgentStrategy) -> list[str]:
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            return []
        parts = [item.strip(" ，。；;？?！!") for item in re.split(r"[；;。]|(?:并且)|(?:以及)|(?:同时)|(?:另外)", normalized)]
        parts = [item for item in parts if item]
        if not parts:
            return [normalized]
        max_subproblems = 2 if strategy == "speed" else 4
        return parts[:max_subproblems]

    def build_plan(
        self,
        *,
        query: str,
        effective_features: list[FeatureFlag],
        route_label: str,
        request: ChatRequest,
        strategy: AgentStrategy,
    ) -> list[PlanStep]:
        steps: list[PlanStep] = [PlanStep("recall_memory", "读取会话记忆")]
        if "rag" in effective_features:
            steps.append(PlanStep("local_rag_search", "本地 RAG 检索"))
        if self._should_use_mcp(query=query, route_label=route_label, strategy=strategy):
            steps.append(PlanStep("mcp_discover", "查看 MCP 工具目录"))
            steps.append(PlanStep("mcp_execute", "尝试执行 MCP 工具"))
        if route_label == "time_sensitive" and "web_search" in effective_features:
            steps.append(PlanStep("official_web_search", "官方联网搜索"))
            steps.append(PlanStep("official_web_read", "官方网页阅读"))
        elif strategy == "quality" and "web_search" in effective_features and self.gateway._is_time_sensitive_query(query):
            steps.append(PlanStep("official_web_search", "官方联网搜索"))
        if route_label == "process":
            if "use_saved_skill" in effective_features and request.saved_skill_id:
                steps.append(PlanStep("saved_skill", "执行历史技能"))
            elif "skill_exec" in effective_features:
                steps.append(PlanStep("general_skill", "执行通用技能"))
        if "citation_guard" in effective_features:
            steps.append(PlanStep("citation_guard", "引用校验"))
        steps.append(PlanStep("synthesize_step", "汇总当前子问题"))
        return steps

    def execute_plan_step(
        self,
        *,
        step: PlanStep,
        subproblem: SubproblemState,
        request: ChatRequest,
        fail_features: set[str],
        effective_features: list[FeatureFlag],
        memory_context_blocks: list[str],
        trace_id: str,
    ) -> StepExecutionResult:
        step_type = step.step_type
        if step_type == "recall_memory":
            _, memory_text, notes = self.load_memory_context(request.session_id)
            return StepExecutionResult(ok=bool(memory_text.strip()), message=memory_text, notes=notes)
        if step_type == "local_rag_search":
            if "rag" not in effective_features:
                return StepExecutionResult(ok=False, message="当前会话未开启 rag 功能。")
            rag_result = self.deps.container.isolation.execute(
                "rag-agent-service",
                lambda: self.gateway._invoke_rag(
                    request.session_id,
                    subproblem.query,
                    fail_features,
                    memory_context_blocks + subproblem.context_blocks,
                ),
            )
            if not rag_result.ok or rag_result.value is None:
                return StepExecutionResult(ok=False, message=f"RAG 检索失败：{rag_result.error or 'unknown'}")
            rag_output = rag_result.value
            notes = []
            if rag_output.degrade_reason:
                notes.append(f"RAG 降级：{rag_output.degrade_reason}")
            return StepExecutionResult(
                ok=bool(rag_output.context_blocks),
                message="\n".join(rag_output.context_blocks[:3]) or "未检索到可靠本地资料。",
                context_blocks=rag_output.context_blocks[: self.deps.services.settings.rag_final_top_k],
                sources=self.dedupe_sources(
                    [ChatSource(title=item.title, url=item.url) for item in rag_output.sources],
                    limit=5,
                ),
                notes=notes,
            )
        if step_type == "official_web_search":
            if "web_search" not in effective_features:
                return StepExecutionResult(ok=False, message="当前会话未开启 web_search 功能。")
            allowed, guarded_query, reason = self.gateway._guard_web_search(subproblem.query)
            if not allowed:
                return StepExecutionResult(ok=False, message=f"联网搜索被拦截：{reason}", tool_audit=[f"web_search:blocked:{reason}"])
            search_result = self.deps.container.isolation.execute(
                "web-search-service",
                lambda: self.gateway._invoke_web_search(guarded_query, fail_features),
            )
            if not search_result.ok or not search_result.value:
                return StepExecutionResult(ok=False, message=f"联网搜索失败：{search_result.error or 'unknown'}")
            hits = search_result.value
            return StepExecutionResult(
                ok=True,
                message="\n".join(f"{item.title}: {item.snippet}" for item in hits[:2]),
                context_blocks=[f"联网搜索摘要：{item.title} | {item.snippet}" for item in hits],
                sources=self.dedupe_sources([ChatSource(title=item.title, url=item.url) for item in hits], limit=5),
                tool_audit=["web_search:allowed:official_whitelist"],
                web_hits=[{"title": item.title, "url": item.url, "snippet": item.snippet} for item in hits],
            )
        if step_type == "official_web_read":
            if not subproblem.web_hits:
                return StepExecutionResult(ok=False, message="当前没有可读的官方网页结果。")
            hits = [self.gateway.WebSearchHit(**item) for item in subproblem.web_hits] if hasattr(self.gateway, "WebSearchHit") else []
            if not hits:
                hits = [type("Hit", (), item) for item in subproblem.web_hits]
            read_result = self.deps.container.isolation.execute(
                "web-read-service",
                lambda: self.gateway._invoke_web_read(query=subproblem.query, hits=hits, fail_features=fail_features),
            )
            if not read_result.ok or not read_result.value:
                return StepExecutionResult(
                    ok=False,
                    message=f"网页阅读失败：{read_result.error or 'unknown'}",
                    tool_audit=["web_read:degraded:official_whitelist"],
                )
            return StepExecutionResult(
                ok=True,
                message="\n".join(read_result.value[:2]),
                context_blocks=read_result.value,
                tool_audit=["web_read:allowed:official_whitelist"],
            )
        if step_type == "general_skill":
            allowed, reason = self.gateway._guard_skill_request(query=subproblem.query, saved_skill_id=None)
            if not allowed:
                return StepExecutionResult(ok=False, message=f"技能执行被拦截：{reason}", tool_audit=[f"skill_exec:blocked:{reason}"])
            skill_result = self.deps.container.isolation.execute(
                "skill-service",
                lambda: self.gateway._invoke_skill(subproblem.query, request.session_id, None, fail_features),
            )
            if not skill_result.ok or not skill_result.value:
                return StepExecutionResult(ok=False, message=f"技能执行失败：{skill_result.error or 'unknown'}")
            return StepExecutionResult(
                ok=True,
                message=str(skill_result.value),
                context_blocks=[f"[skill] {skill_result.value}"],
                tool_audit=[f"skill_exec:allowed:{reason}"],
            )
        if step_type == "saved_skill":
            if not request.saved_skill_id:
                return StepExecutionResult(ok=False, message="当前没有可用的历史技能。")
            allowed, reason = self.gateway._guard_skill_request(query=subproblem.query, saved_skill_id=request.saved_skill_id)
            if not allowed:
                return StepExecutionResult(ok=False, message=f"历史技能调用被拦截：{reason}", tool_audit=[f"use_saved_skill:blocked:{reason}"])
            skill_result = self.deps.container.isolation.execute(
                "saved-skill-service",
                lambda: self.gateway._invoke_skill(subproblem.query, request.session_id, request.saved_skill_id, fail_features),
            )
            if not skill_result.ok or not skill_result.value:
                return StepExecutionResult(ok=False, message=f"历史技能执行失败：{skill_result.error or 'unknown'}")
            return StepExecutionResult(
                ok=True,
                message=str(skill_result.value),
                context_blocks=[f"[saved-skill] {skill_result.value}"],
                tool_audit=[f"use_saved_skill:allowed:{reason}"],
            )
        if step_type == "mcp_discover":
            runtime = self.get_mcp_runtime(trace_id)
            notes = list(runtime.notes)
            if runtime.tools:
                message = "\n".join(f"- {getattr(tool, 'name', 'unknown_tool')}" for tool in runtime.tools[:8])
                return StepExecutionResult(
                    ok=True,
                    message=message or "当前没有可用的 MCP 工具。",
                    tool_audit=[f"agent_tool:mcp_runtime:{','.join(item.alias for item in runtime.servers)}"],
                    notes=notes,
                )
            if runtime.servers:
                return StepExecutionResult(
                    ok=False,
                    message="MCP 服务已配置但工具不可用。",
                    tool_audit=[f"agent_tool:mcp_runtime_unavailable:{','.join(item.alias for item in runtime.servers)}"],
                    notes=notes,
                )
            return StepExecutionResult(ok=False, message="当前未配置 MCP 工具。", notes=notes)
        if step_type == "mcp_execute":
            runtime = self.get_mcp_runtime(trace_id)
            if not runtime.tools:
                return StepExecutionResult(ok=False, message="当前没有可执行的 MCP 工具。", notes=runtime.notes)
            tool = self._select_mcp_tool(runtime.tools, subproblem.query)
            with self._get_mcp_runtime_lock(trace_id):
                try:
                    result = asyncio.run(self._invoke_mcp_tool(tool, subproblem.query))
                except Exception as exc:
                    return StepExecutionResult(ok=False, message=f"MCP 工具执行失败：{exc.__class__.__name__}", notes=runtime.notes)
            return StepExecutionResult(
                ok=bool(str(result).strip()),
                message=str(result).strip() or "MCP 工具未返回内容。",
                context_blocks=[f"[mcp] {str(result).strip()}"] if str(result).strip() else [],
                tool_audit=[f"agent_tool:mcp_execute:{getattr(tool, 'name', 'unknown_tool')}"],
                notes=runtime.notes,
            )
        if step_type == "citation_guard":
            guard_result = self.deps.container.isolation.execute(
                "citation-guard",
                lambda: self.gateway._invoke_citation_guard(self.dedupe_sources(subproblem.sources, 5), fail_features),
            )
            if guard_result.ok and guard_result.value:
                return StepExecutionResult(ok=True, message="引用校验通过。")
            return StepExecutionResult(ok=False, message="引用校验失败或证据不足。")
        if step_type == "synthesize_step":
            chunks = [
                *(f"[step]{value}" for value in subproblem.step_outputs.values() if value.strip()),
                *(f"[note]{item}" for item in subproblem.notes if item.strip()),
            ]
            return StepExecutionResult(
                ok=bool(chunks),
                message="\n".join(chunks[:6]) if chunks else "当前步骤缺少可汇总内容。",
                context_blocks=chunks[:6],
            )
        return StepExecutionResult(ok=False, message=f"未支持的计划步骤：{step_type}")

    async def _invoke_mcp_tool(self, tool: Any, query: str) -> Any:
        if hasattr(tool, "ainvoke"):
            try:
                return await tool.ainvoke(query)
            except Exception:
                return await tool.ainvoke({"query": query})
        if hasattr(tool, "invoke"):
            try:
                return tool.invoke(query)
            except Exception:
                return tool.invoke({"query": query})
        raise RuntimeError("mcp_tool_not_invokable")

    def review_step(self, step: PlanStep, result: StepExecutionResult) -> StepReviewResult:
        if not result.ok:
            return StepReviewResult(ok=False, message=result.message or "步骤执行未返回有效结果。")
        if step.step_type in {"local_rag_search", "official_web_search", "official_web_read"} and not result.context_blocks:
            return StepReviewResult(ok=False, message="步骤缺少可用证据。")
        if step.step_type == "citation_guard" and "通过" not in result.message:
            return StepReviewResult(ok=False, message=result.message or "引用校验失败。")
        if not (result.message or "").strip():
            return StepReviewResult(ok=False, message="步骤执行结果为空。")
        return StepReviewResult(ok=True, message="步骤满足要求。")

    def replan_subproblem(self, subproblem: SubproblemState, request: ChatRequest) -> SubproblemState:
        route_label, _ = self.gateway._classify_query_intent(subproblem.query)
        replanned = SubproblemState(
            subproblem_id=subproblem.subproblem_id,
            query=subproblem.query,
            plan_steps=self.build_plan(
                query=subproblem.query,
                effective_features=request.features,
                route_label=route_label,
                request=request,
                strategy="quality",
            ),
            replan_count=subproblem.replan_count + 1,
        )
        if all(step.step_type != "official_web_read" for step in replanned.plan_steps) and "web_search" in request.features:
            replanned.plan_steps.insert(-1, PlanStep("official_web_read", "补充官方网页阅读"))
        return replanned

    def build_final_session(
        self,
        *,
        request: ChatRequest,
        trace_id: str,
        last_user: str,
        final_text: str,
        sources: list[ChatSource],
        tool_audit: list[str],
        degraded_features: list[FeatureFlag],
        step_events: list[AgentStepEvent],
        status_override: str | None = None,
        error_message: str | None = None,
    ) -> SessionResult:
        prefix_text, degraded = self.gateway._build_citation_notice(request.features, sources, degraded_features)
        merged_text = f"{prefix_text}{final_text}"
        output_flagged, output_reason, audited_text = self.audit_generated_output(merged_text)
        status = status_override or "ok"
        if degraded and status == "ok":
            status = "degraded"
        if output_flagged:
            tool_audit = [*tool_audit, f"safety_audit:output_sanitized:{output_reason}"]
            merged_text = audited_text
            status = "degraded"
        self.gateway._persist_memory_side_effects(request.session_id, last_user, merged_text)
        return SessionResult(
            session_id=request.session_id,
            trace_id=trace_id,
            text=merged_text,
            status=status,  # type: ignore[arg-type]
            degraded_features=list(dict.fromkeys(degraded)),
            sources=self.dedupe_sources(sources, limit=5),
            tool_audit=list(dict.fromkeys(tool_audit)),
            error_message=error_message,
            agent_strategy=request.agent_strategy,
            agent_trace=list(step_events),
        )

    def generate_answer(
        self,
        *,
        request: ChatRequest,
        query: str,
        context_blocks: list[str],
        feature_notes: list[str],
        fail_features: set[str],
    ) -> str:
        generation_output = self.gateway._invoke_generation(
            user_query=query,
            context_blocks=context_blocks,
            feature_notes=feature_notes,
            request=request,
            fail_features=fail_features,
        )
        return generation_output.text

    def should_degrade_generation_error(self, error_message: str | None) -> bool:
        normalized = (error_message or "").strip().lower()
        if not normalized:
            return False
        soft_tokens = (
            "timed out",
            "timeout",
            "circuit_open:generation-service",
        )
        hard_tokens = (
            "generation failure injected",
        )
        if any(token in normalized for token in hard_tokens):
            return False
        return any(token in normalized for token in soft_tokens)

    def compact_generation_context(self, *, context_blocks: list[str], strategy: AgentStrategy) -> list[str]:
        limit = 4 if strategy == "speed" else 6
        compacted: list[str] = []
        seen: set[str] = set()
        for item in context_blocks:
            normalized = self._compact_text_block(item, limit_chars=220)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            compacted.append(normalized)
            if len(compacted) >= limit:
                break
        return compacted

    def compact_feature_notes(self, *, notes: list[str], strategy: AgentStrategy) -> list[str]:
        limit = 6 if strategy == "speed" else 8
        compacted: list[str] = []
        seen: set[str] = set()
        for item in notes:
            normalized = self._compact_text_block(item, limit_chars=120)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            compacted.append(normalized)
            if len(compacted) >= limit:
                break
        return compacted

    def build_rule_based_final_answer(
        self,
        *,
        query: str,
        context_blocks: list[str],
        notes: list[str],
        error_message: str,
    ) -> str:
        rows = [
            "当前最终生成阶段超时，以下为基于已完成步骤的保守汇总：",
            f"问题：{query}",
            "",
        ]
        evidence_rows = [item.strip() for item in context_blocks if item.strip()][:5]
        note_rows = [item.strip() for item in notes if item.strip()][:5]
        if evidence_rows:
            rows.append("已取得的依据：")
            rows.extend(f"- {item[:180]}" for item in evidence_rows)
            rows.append("")
        if note_rows:
            rows.append("执行备注：")
            rows.extend(f"- {item[:160]}" for item in note_rows)
            rows.append("")
        rows.append(f"失败原因：{error_message}")
        rows.append("建议：稍后重试，或切换更快模型/速度优先策略后再试。")
        return "\n".join(rows).strip()

    def _rule_rewrite_query(self, *, last_user: str, memory_text: str, strategy: AgentStrategy) -> str:
        normalized = " ".join(last_user.split()).strip()
        if strategy == "quality" and memory_text and any(token in normalized for token in ("这个", "那个", "它", "那")):
            first_memory_line = next((line[2:] for line in memory_text.splitlines() if line.startswith("- ")), "")
            if first_memory_line:
                return f"{normalized}（结合上下文：{first_memory_line}）"
        return normalized

    def _compact_text_block(self, text: str, *, limit_chars: int) -> str:
        normalized = re.sub(r"\s+", " ", (text or "")).strip()
        if len(normalized) <= limit_chars:
            return normalized
        keep = normalized[: limit_chars - 3].rstrip(" ，。；;,:：")
        return f"{keep}..."

    def _should_use_mcp(self, *, query: str, route_label: str, strategy: AgentStrategy) -> bool:
        servers, _ = load_mcp_server_configs(self.deps.services.settings)
        if not servers:
            return False

        normalized = query.lower()
        if any(token in normalized for token in ("mcp", "外部工具", "模型上下文协议", "bing", "fetch")):
            return True

        capability_text = " ".join(
            " ".join(
                [
                    item.alias,
                    item.original_name,
                    item.command,
                    *item.args,
                    item.url,
                ]
            ).lower()
            for item in servers
        )
        has_search_capability = any(token in capability_text for token in ("search", "bing", "serp", "query"))
        has_fetch_capability = any(token in capability_text for token in ("fetch", "crawl", "read", "browser", "web"))
        if has_search_capability and (
            route_label == "time_sensitive" or any(token in query for token in ("搜索", "查询", "查一下", "搜一下", "最新", "公告", "官网", "官方"))
        ):
            return True
        if has_fetch_capability and any(token in query for token in ("网页", "页面", "链接", "抓取", "读取", "打开")):
            return True
        return strategy == "quality" and route_label == "time_sensitive" and (has_search_capability or has_fetch_capability)

    def get_mcp_runtime(self, trace_id: str) -> McpToolRuntime:
        with self._mcp_runtime_guard:
            cached = self._mcp_runtime_cache.get(trace_id)
            if cached is not None:
                self.logger.info("[agent.mcp] trace=%s cache_hit tools=%s", trace_id, len(cached.tools))
                return cached

        started_at = perf_counter()
        runtime = asyncio.run(build_langchain_mcp_runtime(self.deps.services.settings))
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        self.logger.info(
            "[agent.mcp] trace=%s runtime_ready elapsed_ms=%s tools=%s servers=%s",
            trace_id,
            elapsed_ms,
            len(runtime.tools),
            len(runtime.servers),
        )
        with self._mcp_runtime_guard:
            cached = self._mcp_runtime_cache.get(trace_id)
            if cached is not None:
                started_at = perf_counter()
                asyncio.run(runtime.aclose())
                close_elapsed_ms = int((perf_counter() - started_at) * 1000)
                self.logger.info(
                    "[agent.mcp] trace=%s duplicate_runtime_closed elapsed_ms=%s",
                    trace_id,
                    close_elapsed_ms,
                )
                return cached
            self._mcp_runtime_cache[trace_id] = runtime
            self._mcp_runtime_locks.setdefault(trace_id, threading.RLock())
            return runtime

    def release_mcp_runtime(self, trace_id: str) -> None:
        runtime: McpToolRuntime | None = None
        with self._mcp_runtime_guard:
            runtime = self._mcp_runtime_cache.pop(trace_id, None)
            self._mcp_runtime_locks.pop(trace_id, None)
        if runtime is None:
            return
        started_at = perf_counter()
        asyncio.run(runtime.aclose())
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        self.logger.info("[agent.mcp] trace=%s runtime_released elapsed_ms=%s", trace_id, elapsed_ms)

    def _get_mcp_runtime_lock(self, trace_id: str) -> threading.RLock:
        with self._mcp_runtime_guard:
            lock = self._mcp_runtime_locks.get(trace_id)
            if lock is None:
                lock = threading.RLock()
                self._mcp_runtime_locks[trace_id] = lock
            return lock

    def _select_mcp_tool(self, tools: list[Any], query: str) -> Any:
        if len(tools) == 1:
            return tools[0]

        normalized_query = query.lower()
        query_prefers_search = any(token in query for token in ("搜索", "查询", "查一下", "搜一下", "最新", "公告", "官网", "官方"))
        query_prefers_fetch = any(token in query for token in ("网页", "页面", "链接", "抓取", "读取", "打开"))

        def _score(tool: Any) -> tuple[int, str]:
            name = str(getattr(tool, "name", "") or "").lower()
            description = str(getattr(tool, "description", "") or "").lower()
            haystack = f"{name} {description}"
            score = 0
            if "search" in normalized_query or query_prefers_search:
                if any(token in haystack for token in ("search", "bing", "query", "web_search")):
                    score += 3
            if query_prefers_fetch:
                if any(token in haystack for token in ("fetch", "read", "crawl", "browser", "web_read")):
                    score += 3
            if "mcp" in haystack:
                score += 1
            return score, name

        return max(tools, key=_score)

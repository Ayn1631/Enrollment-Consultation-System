from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import quote

from app.contracts import GenerationRequest, MemoryEntry
from app.models import (
    ChatCreateResponse,
    ChatRequest,
    ChatSource,
    ChatStatus,
    FeatureFlag,
    SessionResult,
)
from app.services.service_client import ServiceClient
from app.state import ServiceContainer


@dataclass(slots=True)
class GatewayDependencies:
    container: ServiceContainer
    services: ServiceClient


@dataclass(slots=True)
class WebSearchHit:
    title: str
    url: str
    snippet: str


@dataclass(slots=True)
class QueryRouteDecision:
    route_label: str
    reason: str
    features: list[FeatureFlag]
    notes: list[str]
    audit: list[str]


@dataclass(slots=True)
class PreparedChatContext:
    trace_id: str
    session_id: str
    last_user: str
    effective_features: list[FeatureFlag]
    degraded: list[FeatureFlag]
    feature_notes: list[str]
    sources: list[ChatSource]
    context_blocks: list[str]
    tool_audit: list[str]
    blocked_reply: str | None = None


@dataclass(slots=True)
class GatewayStreamEvent:
    event: str
    data: dict[str, Any]


class GatewayOrchestrator:
    WEB_SEARCH_ALLOWED_DOMAINS: tuple[str, ...] = ("zsc.zut.edu.cn", "zut.edu.cn")
    logger = logging.getLogger(__name__)

    def __init__(self, deps: GatewayDependencies):
        self.deps = deps

    def create_chat(self, request: ChatRequest, fail_features: set[str] | None = None) -> ChatCreateResponse:
        """网关主流程：按 Agent 规划顺序执行功能并统一处理降级。"""
        fail_features = fail_features or set()
        prepared = self._prepare_chat_context(request=request, fail_features=fail_features)
        if prepared.blocked_reply is not None:
            session = self._build_blocked_session(prepared)
            self.deps.container.session_store.set(request.session_id, session)
            return self._to_create_response(session)

        generation_result = self.deps.container.isolation.execute(
            "generation-service",
            lambda: self._invoke_generation(
                user_query=prepared.last_user,
                context_blocks=prepared.context_blocks,
                feature_notes=prepared.feature_notes,
                request=request,
                fail_features=fail_features,
            ),
        )
        print(
            f"[Gateway] generation_result trace_id={prepared.trace_id} ok={generation_result.ok} "
            f"error={generation_result.error} degraded={generation_result.degraded} "
            f"context_blocks={len(prepared.context_blocks)} sources={len(prepared.sources)}"
        )
        if not generation_result.ok or generation_result.value is None:
            self.logger.error(
                "generation failed trace_id=%s session_id=%s error=%s features=%s",
                prepared.trace_id,
                request.session_id,
                generation_result.error or "generation failed",
                prepared.effective_features,
            )
            session = self._build_failed_generation_session(
                prepared=prepared,
                error_message=generation_result.error or "generation failed",
            )
            self.deps.container.session_store.set(request.session_id, session)
            return self._to_create_response(session)

        session = self._build_success_session(
            prepared=prepared,
            generation_output=generation_result.value,
        )
        self.deps.container.session_store.set(request.session_id, session)
        return self._to_create_response(session)

    def stream_chat(self, request: ChatRequest, fail_features: set[str] | None = None) -> Iterator[GatewayStreamEvent]:
        """单请求流式聊天：前置能力准备完成后，边生成边向前端输出增量。"""
        fail_features = fail_features or set()
        prepared = self._prepare_chat_context(request=request, fail_features=fail_features)
        if prepared.blocked_reply is not None:
            session = self._build_blocked_session(prepared)
            self.deps.container.session_store.set(request.session_id, session)
            yield from self._yield_text_events(session.text)
            yield self._build_done_event(session)
            return

        prefix_text, degraded = self._build_citation_notice(
            effective_features=prepared.effective_features,
            sources=prepared.sources,
            degraded=prepared.degraded,
        )
        emitted_parts: list[str] = []
        if prefix_text:
            emitted_parts.append(prefix_text)
            yield from self._yield_text_events(prefix_text)

        generation_output = None
        generation_error: str | None = None
        for item in self.deps.container.isolation.execute_stream(
            "generation-service",
            lambda: self._invoke_generation_stream(
                user_query=prepared.last_user,
                context_blocks=prepared.context_blocks,
                feature_notes=prepared.feature_notes,
                request=request,
                fail_features=fail_features,
            ),
        ):
            if not item.ok:
                generation_error = item.error or "generation failed"
                break
            chunk = item.value
            if chunk is None:
                continue
            if chunk.done:
                generation_output = chunk.response
                continue
            if chunk.delta:
                emitted_parts.append(chunk.delta)
                yield GatewayStreamEvent(event="message", data={"delta": chunk.delta})

        if generation_output is None:
            failure_text = ""
            if not emitted_parts:
                failure_text = "当前生成服务异常，请稍后重试。"
                yield from self._yield_text_events(failure_text)
            else:
                failure_text = "".join(emitted_parts) + "\n\n生成过程中断，请稍后重试。"
                yield from self._yield_text_events("\n\n生成过程中断，请稍后重试。")
            session = self._build_failed_generation_session(
                prepared=prepared,
                error_message=generation_error or "generation failed",
                text_override=failure_text,
                degraded_override=degraded,
            )
            self.deps.container.session_store.set(request.session_id, session)
            yield self._build_done_event(session)
            return

        session = self._build_success_session(
            prepared=prepared,
            generation_output=generation_output,
            prefix_override=prefix_text,
            degraded_override=degraded,
        )
        self.deps.container.session_store.set(request.session_id, session)
        yield self._build_done_event(session)

    def _route_features(self, query: str, request: ChatRequest) -> QueryRouteDecision:
        """按问题类型动态裁剪工具链"""
        route_label, reason = self._classify_query_intent(query)
        routed = list(dict.fromkeys(request.features))
        notes: list[str] = []
        audit = [f"query_router:label:{route_label}:{reason}"]

        if route_label == "time_sensitive" and "rag" in routed and "web_search" not in routed:
            routed.append("web_search")
            notes.append("Query Router 识别为时效问题，已自动开启联网搜索增强。")
            audit.append("query_router:auto_enable:web_search")

        if route_label == "process" and "use_saved_skill" not in routed and "skill_exec" not in routed:
            routed.append("skill_exec")
            notes.append("Query Router 识别为流程咨询，已自动开启技能执行链路。")
            audit.append("query_router:auto_enable:skill_exec")

        if route_label == "follow_up" and "web_search" in routed:
            routed = [feature for feature in routed if feature != "web_search"]
            notes.append("Query Router 识别为追问，已关闭联网搜索并优先复用记忆与本地检索。")
            audit.append("query_router:auto_disable:web_search")

        if route_label == "smalltalk":
            removable = [feature for feature in routed if feature in {"web_search", "skill_exec", "use_saved_skill"}]
            if removable:
                routed = [feature for feature in routed if feature not in {"web_search", "skill_exec", "use_saved_skill"}]
                notes.append("Query Router 识别为闲聊，已关闭外部工具链路。")
                audit.append(f"query_router:auto_disable:{'+'.join(removable)}")

        return QueryRouteDecision(
            route_label=route_label,
            reason=reason,
            features=routed,
            notes=notes,
            audit=audit,
        )

    def _invoke_rag(
        self,
        session_id: str,
        query: str,
        fail_features: set[str],
        memory_context_blocks: list[str],
    ):
        """执行 LangGraph RAG 调用，支持测试注入 rag 故障。"""
        if "rag" in fail_features:
            raise RuntimeError("rag failure injected")
        return self.deps.services.run_rag_graph(
            session_id=session_id,
            query=query,
            top_k=self.deps.services.settings.rag_final_top_k,
            debug=True,
            memory_context_blocks=memory_context_blocks,
        )

    def _invoke_web_search(self, query: str, fail_features: set[str]) -> list[WebSearchHit]:
        """执行联网搜索补充，限制为官方域名并返回候选网页。"""
        if "web_search" in fail_features:
            raise RuntimeError("web search failure injected")
        encoded_query = quote(query)
        return [
            WebSearchHit(
                title=f"中原工学院官方结果：{query}",
                url=f"https://{self.WEB_SEARCH_ALLOWED_DOMAINS[0]}/search?keyword={encoded_query}",
                snippet=f"仅允许参考 {self.WEB_SEARCH_ALLOWED_DOMAINS[0]} 与 {self.WEB_SEARCH_ALLOWED_DOMAINS[1]} 的官方最新通知。",
            )
        ]

    def _invoke_web_read(self, query: str, hits: list[WebSearchHit], fail_features: set[str]) -> list[str]:
        """对官方搜索结果执行网页阅读，提取可入模的摘要。"""
        if "web_search" in fail_features or "web_read" in fail_features:
            raise RuntimeError("web read failure injected")
        blocks: list[str] = []
        for item in hits[:2]:
            blocks.append(
                f"[official-page][query={query}][title={item.title}][url={item.url}]\n"
                f"{item.snippet}"
            )
        return blocks

    def _invoke_skill(
        self,
        query: str,
        session_id: str,
        saved_skill_id: str | None,
        fail_features: set[str],
    ) -> str:
        """执行技能调用，按是否指定 saved_skill_id 选择执行路径。"""
        if saved_skill_id and "use_saved_skill" in fail_features:
            raise RuntimeError("saved skill failure injected")
        if not saved_skill_id and "skill_exec" in fail_features:
            raise RuntimeError("skill failure injected")
        result = self.deps.services.execute_skill(
            query=query,
            session_id=session_id,
            saved_skill_id=saved_skill_id,
        )
        return result.note

    def _invoke_citation_guard(self, sources: list[ChatSource], fail_features: set[str]) -> bool:
        """执行引用校验，失败时由外层降级并切换保守模板。"""
        if "citation_guard" in fail_features:
            raise RuntimeError("citation guard failure injected")
        result = self.deps.services.citation_guard(sources)
        return result.ok

    def _invoke_generation(
        self,
        user_query: str,
        context_blocks: list[str],
        feature_notes: list[str],
        request: ChatRequest,
        fail_features: set[str],
    ):
        """执行最终生成，generation 失败属于硬失败。"""
        if "generation" in fail_features:
            raise RuntimeError("generation failure injected")
        return self.deps.services.generate(
            GenerationRequest(
                user_query=user_query,
                context_blocks=context_blocks,
                feature_notes=feature_notes,
                model=request.model,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        )

    def _invoke_generation_stream(
        self,
        user_query: str,
        context_blocks: list[str],
        feature_notes: list[str],
        request: ChatRequest,
        fail_features: set[str],
    ):
        """执行最终生成的流式版本，供真正的 SSE 接口复用。"""
        if "generation" in fail_features:
            raise RuntimeError("generation failure injected")
        return self.deps.services.stream_generate(
            GenerationRequest(
                user_query=user_query,
                context_blocks=context_blocks,
                feature_notes=feature_notes,
                model=request.model,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        )

    def _prepare_chat_context(self, request: ChatRequest, fail_features: set[str]) -> PreparedChatContext:
        """执行生成前的所有准备步骤，供同步与流式接口复用。"""
        trace_id = uuid.uuid4().hex
        degraded: list[FeatureFlag] = []
        feature_notes: list[str] = []
        sources: list[ChatSource] = []
        context_blocks: list[str] = []
        tool_audit: list[str] = []

        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "").strip()
        if not last_user:
            last_user = "请介绍中原工学院招生政策要点。"
        print(
            f"[Gateway] create_chat start trace_id={trace_id} session_id={request.session_id} "
            f"features={request.features} strict_citation={request.strict_citation} user={last_user[:120]}"
        )
        input_blocked, input_reason, safe_reply = self._audit_user_input(last_user)
        if input_blocked:
            tool_audit.append(f"safety_audit:input_blocked:{input_reason}")
            return PreparedChatContext(
                trace_id=trace_id,
                session_id=request.session_id,
                last_user=last_user,
                effective_features=list(request.features),
                degraded=[],
                feature_notes=feature_notes,
                sources=sources,
                context_blocks=context_blocks,
                tool_audit=tool_audit,
                blocked_reply=safe_reply,
            )

        route_decision = self._route_features(query=last_user, request=request)
        print(
            f"[Gateway] route_decision trace_id={trace_id} label={route_decision.route_label} "
            f"reason={route_decision.reason} features={route_decision.features}"
        )
        tool_audit.extend(route_decision.audit)
        feature_notes.extend(route_decision.notes)
        effective_features = route_decision.features
        ordered_features = self.deps.services.plan_features(effective_features)

        memory_result = self.deps.container.isolation.execute(
            "memory-service",
            lambda: self.deps.services.read_short_memory(request.session_id),
        )
        if memory_result.ok and memory_result.value and memory_result.value.entries:
            context_blocks.extend([f"[memory] {item.value}" for item in memory_result.value.entries[:3]])
            feature_notes.append("短期记忆已接入上下文。")
        elif memory_result.ok:
            feature_notes.append("当前会话暂无短期记忆，已跳过。")
        else:
            feature_notes.append("短期记忆服务不可用，已忽略。")
        self._append_optional_memory_context(
            context_blocks=context_blocks,
            feature_notes=feature_notes,
            session_id=request.session_id,
            kind="long",
            label="长期记忆",
            prefix="[long-memory]",
        )
        self._append_optional_memory_context(
            context_blocks=context_blocks,
            feature_notes=feature_notes,
            session_id=request.session_id,
            kind="special",
            label="特殊记忆",
            prefix="[special-memory]",
        )
        rag_memory_context_blocks = list(context_blocks)

        for feature in ordered_features:
            if feature == "rag":
                rag_result = self.deps.container.isolation.execute(
                    "rag-agent-service",
                    lambda: self._invoke_rag(
                        request.session_id,
                        last_user,
                        fail_features,
                        rag_memory_context_blocks,
                    ),
                )
                print(
                    f"[Gateway] rag_result trace_id={trace_id} ok={rag_result.ok} "
                    f"error={rag_result.error} degraded={rag_result.degraded}"
                )
                if rag_result.ok and rag_result.value is not None:
                    rag_output = rag_result.value
                    context_blocks.extend(rag_output.context_blocks[: self.deps.services.settings.rag_final_top_k])
                    sources = self._dedupe_chat_sources(
                        [ChatSource(title=item.title, url=item.url) for item in rag_output.sources],
                        limit=5,
                    )
                    if rag_output.status == "degraded":
                        print(
                            f"[Gateway] rag_output degraded trace_id={trace_id} "
                            f"reason={rag_output.degrade_reason} sources={len(rag_output.sources)}"
                        )
                        if rag_output.degrade_reason and rag_output.degrade_reason.startswith("node_timeout:") and rag_output.sources:
                            feature_notes.append(f"RAG 节点耗时偏高：{rag_output.degrade_reason}，已保留有效检索证据。")
                        else:
                            degraded.append("rag")
                            if rag_output.degrade_reason:
                                feature_notes.append(f"RAG 降级：{rag_output.degrade_reason}")
                    else:
                        print(
                            f"[Gateway] rag_output ok trace_id={trace_id} "
                            f"context_blocks={len(rag_output.context_blocks)} sources={len(sources)}"
                        )
                        feature_notes.append("RAG LangGraph 工作流执行成功。")
                else:
                    degraded.append("rag")
                    feature_notes.append("RAG 检索失败，降级为无检索回答。")
                continue

            if feature == "web_search":
                allowed, guarded_query, reason = self._guard_web_search(last_user)
                tool_audit.append(f"web_search:{'allowed' if allowed else 'blocked'}:{reason}")
                if not allowed:
                    degraded.append("web_search")
                    feature_notes.append(f"联网搜索已拦截：{reason}")
                    continue
                web_result = self.deps.container.isolation.execute(
                    "web-search-service",
                    lambda: self._invoke_web_search(guarded_query, fail_features),
                )
                if web_result.ok and web_result.value:
                    hits = web_result.value
                    context_blocks.extend([f"联网搜索摘要：{item.title} | {item.snippet}" for item in hits])
                    read_result = self.deps.container.isolation.execute(
                        "web-read-service",
                        lambda: self._invoke_web_read(query=guarded_query, hits=hits, fail_features=fail_features),
                    )
                    if read_result.ok and read_result.value:
                        tool_audit.append("web_read:allowed:official_whitelist")
                        context_blocks.extend(read_result.value)
                        feature_notes.append("联网搜索与官方网页阅读补充成功。")
                    else:
                        tool_audit.append("web_read:degraded:official_whitelist")
                        degraded.append("web_search")
                        feature_notes.append("官方网页阅读失败，已保留搜索摘要并标记降级。")
                else:
                    degraded.append("web_search")
                    feature_notes.append("联网搜索失败，已降级。")
                continue

            if feature == "skill_exec":
                allowed, reason = self._guard_skill_request(query=last_user, saved_skill_id=None)
                tool_audit.append(f"skill_exec:{'allowed' if allowed else 'blocked'}:{reason}")
                if not allowed:
                    degraded.append("skill_exec")
                    feature_notes.append(f"技能执行已拦截：{reason}")
                    continue
                skill_result = self.deps.container.isolation.execute(
                    "skill-service",
                    lambda: self._invoke_skill(last_user, request.session_id, None, fail_features),
                )
                if skill_result.ok and skill_result.value:
                    feature_notes.append(skill_result.value)
                else:
                    degraded.append("skill_exec")
                    feature_notes.append("技能执行失败，已跳过。")
                continue

            if feature == "use_saved_skill":
                allowed, reason = self._guard_skill_request(query=last_user, saved_skill_id=request.saved_skill_id)
                tool_audit.append(f"use_saved_skill:{'allowed' if allowed else 'blocked'}:{reason}")
                if not allowed:
                    degraded.append("use_saved_skill")
                    feature_notes.append(f"历史技能调用已拦截：{reason}")
                    continue
                saved_skill_result = self.deps.container.isolation.execute(
                    "saved-skill-service",
                    lambda: self._invoke_skill(last_user, request.session_id, request.saved_skill_id, fail_features),
                )
                if saved_skill_result.ok and saved_skill_result.value:
                    feature_notes.append(saved_skill_result.value)
                else:
                    degraded.append("use_saved_skill")
                    feature_notes.append("历史技能不可用，已回退通用流程。")
                continue

            if feature == "citation_guard":
                guard_result = self.deps.container.isolation.execute(
                    "citation-guard",
                    lambda: self._invoke_citation_guard(sources=sources, fail_features=fail_features),
                )
                print(
                    f"[Gateway] citation_guard trace_id={trace_id} ok={guard_result.ok} "
                    f"value={guard_result.value} error={guard_result.error} sources={len(sources)}"
                )
                if guard_result.ok and guard_result.value:
                    feature_notes.append("引用校验通过。")
                elif sources and self._can_soft_pass_citation_guard(guard_result=guard_result):
                    tool_audit.append(f"citation_guard:soft_pass:{guard_result.error or 'service_unavailable'}")
                    feature_notes.append("引用校验服务异常，但已检测到可展示来源，已按保守策略继续回答。")
                else:
                    degraded.append("citation_guard")
                    feature_notes.append("引用校验失败，已启用保守模板。")

        return PreparedChatContext(
            trace_id=trace_id,
            session_id=request.session_id,
            last_user=last_user,
            effective_features=effective_features,
            degraded=degraded,
            feature_notes=feature_notes,
            sources=sources,
            context_blocks=context_blocks,
            tool_audit=tool_audit,
        )

    def _build_blocked_session(self, prepared: PreparedChatContext) -> SessionResult:
        return SessionResult(
            session_id=prepared.session_id,
            trace_id=prepared.trace_id,
            text=prepared.blocked_reply or "",
            status="degraded",
            degraded_features=[],
            sources=[],
            tool_audit=prepared.tool_audit,
            finish_reason="stop",
        )

    def _build_failed_generation_session(
        self,
        prepared: PreparedChatContext,
        error_message: str,
        text_override: str | None = None,
        degraded_override: list[FeatureFlag] | None = None,
    ) -> SessionResult:
        degraded = list(dict.fromkeys(degraded_override if degraded_override is not None else prepared.degraded))
        return SessionResult(
            session_id=prepared.session_id,
            trace_id=prepared.trace_id,
            text=text_override or "当前生成服务异常，请稍后重试。",
            status="failed",
            degraded_features=degraded,
            sources=prepared.sources,
            tool_audit=prepared.tool_audit,
            finish_reason="error",
            error_message=error_message,
        )

    def _build_success_session(
        self,
        prepared: PreparedChatContext,
        generation_output,
        prefix_override: str | None = None,
        degraded_override: list[FeatureFlag] | None = None,
    ) -> SessionResult:
        tool_audit = list(prepared.tool_audit)
        tool_audit.append(
            "generation:"
            f"{generation_output.route}:"
            f"{generation_output.model or 'unknown'}:"
            f"cache_{'hit' if generation_output.cache_hit else 'miss'}"
        )
        degraded = list(degraded_override if degraded_override is not None else prepared.degraded)
        prefix_text = prefix_override
        if prefix_text is None:
            prefix_text, degraded = self._build_citation_notice(
                effective_features=prepared.effective_features,
                sources=prepared.sources,
                degraded=degraded,
            )
        final_text = f"{prefix_text}{generation_output.text}"
        output_flagged, output_reason, audited_text = self._audit_generated_output(final_text)
        status: ChatStatus = "ok"
        if output_flagged:
            tool_audit.append(f"safety_audit:output_sanitized:{output_reason}")
            final_text = audited_text
            status = "degraded"
        if degraded:
            status = "degraded"
        print(
            f"[Gateway] create_chat done trace_id={prepared.trace_id} status={status} "
            f"degraded={list(dict.fromkeys(degraded))} sources={len(prepared.sources)} tool_audit={tool_audit}"
        )
        self._persist_memory_side_effects(session_id=prepared.session_id, last_user=prepared.last_user, final_text=final_text)
        return SessionResult(
            session_id=prepared.session_id,
            trace_id=prepared.trace_id,
            text=final_text,
            status=status,
            degraded_features=list(dict.fromkeys(degraded)),
            sources=prepared.sources,
            tool_audit=tool_audit,
        )

    def _persist_memory_side_effects(self, session_id: str, last_user: str, final_text: str) -> None:
        self.deps.container.isolation.execute(
            "memory-service",
            lambda: self.deps.services.write_short_memory(session_id, "last_user_query", last_user),
        )
        self.deps.container.isolation.execute(
            "memory-service",
            lambda: self.deps.services.append_long_memory_summary(
                session_id,
                self._build_long_memory_snippet(last_user=last_user, response_text=final_text),
            ),
        )
        special_preference = self._infer_special_memory(last_user)
        if special_preference is not None:
            self.deps.container.isolation.execute(
                "memory-service",
                lambda: self.deps.services.write_memory(session_id, special_preference),
            )

    def _build_citation_notice(
        self,
        effective_features: list[FeatureFlag],
        sources: list[ChatSource],
        degraded: list[FeatureFlag],
    ) -> tuple[str, list[FeatureFlag]]:
        deduped_degraded = list(dict.fromkeys(degraded))
        if "citation_guard" not in effective_features:
            return "", deduped_degraded
        if sources and "citation_guard" not in deduped_degraded:
            return "", deduped_degraded
        if "citation_guard" not in deduped_degraded:
            deduped_degraded.append("citation_guard")
        prefix_text = (
            "当前证据链不完整，以下内容仅供参考。\n"
            "建议联系招生办电话 0371-67698700 / 67698712 / 67698674 进一步确认。\n\n"
        )
        return prefix_text, deduped_degraded

    def _yield_text_events(self, text: str) -> Iterator[GatewayStreamEvent]:
        chunk_size = max(1, self.deps.services.settings.stream_chunk_size)
        for idx in range(0, len(text), chunk_size):
            yield GatewayStreamEvent(event="message", data={"delta": text[idx : idx + chunk_size]})

    def _build_done_event(self, session: SessionResult) -> GatewayStreamEvent:
        return GatewayStreamEvent(
            event="done",
            data={
                "finish_reason": session.finish_reason,
                "status": session.status,
                "degraded_features": session.degraded_features,
                "sources": [item.model_dump() for item in session.sources],
                "trace_id": session.trace_id,
                "tool_audit": session.tool_audit,
            },
        )

    def _to_create_response(self, session: SessionResult) -> ChatCreateResponse:
        return ChatCreateResponse(
            session_id=session.session_id,
            trace_id=session.trace_id,
            status=session.status,
            degraded_features=session.degraded_features,
        )

    def _append_optional_memory_context(
        self,
        context_blocks: list[str],
        feature_notes: list[str],
        session_id: str,
        kind: str,
        label: str,
        prefix: str,
    ) -> None:
        """按种类加载非关键记忆，失败时只记备注不打断主流程。"""
        memory_result = self.deps.container.isolation.execute(
            "memory-service",
            lambda: self.deps.services.read_memory(session_id=session_id, kind=kind),
        )
        if memory_result.ok and memory_result.value and memory_result.value.entries:
            context_blocks.extend([f"{prefix} {item.value}" for item in memory_result.value.entries[:2]])
            feature_notes.append(f"{label}已接入上下文。")

    def _dedupe_chat_sources(self, sources: list[ChatSource], limit: int) -> list[ChatSource]:
        """按 url/title 去重来源，避免同一文档不同 chunk 被重复展示。"""
        deduped: list[ChatSource] = []
        seen: set[tuple[str, str]] = set()
        for source in sources:
            key = (source.url.strip(), source.title.strip())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
            if len(deduped) >= limit:
                break
        return deduped

    def _can_soft_pass_citation_guard(self, guard_result) -> bool:
        """已有来源时，引用校验服务自身异常可软通过，避免整轮回答被误伤。"""
        if guard_result.ok:
            return False
        error = (guard_result.error or "").strip().lower()
        soft_errors = ("circuit_open:", "timeout", "connection", "temporarily unavailable")
        return any(token in error for token in soft_errors) or bool(error)

    def _build_long_memory_snippet(self, last_user: str, response_text: str) -> str:
        """构造滚动摘要片段，给长期记忆做增量更新。"""
        answer_excerpt = " ".join(response_text.split())[:160]
        return f"用户关注：{last_user[:80]}；系统回应摘要：{answer_excerpt}"

    def _infer_special_memory(self, last_user: str):
        """从用户表达中提炼稳定偏好，写入 special memory。"""
        preference_map = {
            "简短": "偏好简短回答",
            "简洁": "偏好简短回答",
            "详细": "偏好详细回答",
            "分点": "偏好分点回答",
            "表格": "偏好表格化展示",
        }
        for keyword, value in preference_map.items():
            if keyword in last_user:
                return MemoryEntry(
                    key="response_style",
                    value=value,
                    kind="special",
                    confidence=0.88,
                    source="user_preference",
                )
        return None

    def _guard_web_search(self, query: str) -> tuple[bool, str, str]:
        """联网搜索白名单与参数校验，只放行强时效且长度受控的问题。"""
        normalized = " ".join(query.split()).strip()
        if not normalized:
            return False, normalized, "empty_query"
        if len(normalized) > 120:
            return False, normalized[:120], "query_too_long"
        if not self._is_time_sensitive_query(normalized):
            return False, normalized, "not_time_sensitive"
        cleaned = re.sub(r"[^\w\u4e00-\u9fff\s\-:/\.]", " ", normalized)
        cleaned = " ".join(cleaned.split())
        return True, cleaned, "official_whitelist"

    def _guard_skill_request(self, query: str, saved_skill_id: str | None) -> tuple[bool, str]:
        """技能调用最小权限校验：参数长度和 saved skill 白名单。"""
        normalized = " ".join(query.split()).strip()
        if not normalized:
            return False, "empty_query"
        if len(normalized) > 200:
            return False, "query_too_long"
        if saved_skill_id:
            allowed_ids = {item.id for item in self.deps.services.list_saved_skills().skills}
            if saved_skill_id not in allowed_ids:
                return False, "saved_skill_not_allowed"
            return True, "saved_skill_whitelisted"
        return True, "generic_skill_allowed"

    def _is_time_sensitive_query(self, query: str) -> bool:
        keywords = ("最新", "当前", "现在", "今年", "最近", "近期", "公告", "通知", "今日", "今天")
        return any(keyword in query for keyword in keywords) or bool(re.search(r"\b20\d{2}\b", query))

    def _classify_query_intent(self, query: str) -> tuple[str, str]:
        """识别问题类型，供网关级 Query Router 选择工具链。"""
        normalized = " ".join(query.split()).strip()
        if not normalized:
            return "policy", "empty_query"
        if self._is_time_sensitive_query(normalized):
            return "time_sensitive", "time_sensitive_keyword"
        if len(normalized) <= 14 and any(keyword in normalized for keyword in ("那", "还", "这个", "那个", "呢", "吗", "再说")):
            return "follow_up", "short_follow_up"
        if any(keyword in normalized for keyword in ("你好", "在吗", "谢谢", "哈哈", "hi", "hello")):
            return "smalltalk", "smalltalk_keyword"
        if any(keyword in normalized for keyword in ("流程", "步骤", "怎么", "如何", "办理", "报到", "报名", "申请", "提交材料")):
            return "process", "process_keyword"
        if any(keyword in normalized for keyword in ("电话", "地址", "学费", "住宿", "资助", "奖学金", "贷款", "收费")):
            return "faq", "faq_keyword"
        return "policy", "default_policy"

    def _audit_user_input(self, query: str) -> tuple[bool, str, str]:
        normalized = " ".join(query.split()).strip()
        if not normalized:
            return False, "ok", ""
        rules = [
            (
                r"(?i)(输出|展示|泄露).*(系统提示词|提示词|内部指令|developer message|system prompt)",
                "prompt_leak_request",
            ),
            (
                r"(?i)(忽略|绕过).*(系统|规则|限制|审计|校验)",
                "policy_bypass_request",
            ),
        ]
        for pattern, reason in rules:
            if re.search(pattern, normalized):
                return (
                    True,
                    reason,
                    "该请求涉及系统提示词、内部策略或安全边界，不能直接提供。\n"
                    "如果你是想了解招生政策、流程、学费或资助，我可以继续基于公开资料帮你整理。",
                )
        return False, "ok", ""

    def _audit_generated_output(self, text: str) -> tuple[bool, str, str]:
        normalized = text or ""
        rules = [
            (
                r"(?i)(系统提示词|system prompt|developer message|内部指令)",
                "prompt_leak_output",
            ),
            (
                r"(?i)(api[_\s-]?key|access[_\s-]?token|sk-[a-z0-9]{10,})",
                "secret_like_output",
            ),
        ]
        for pattern, reason in rules:
            if re.search(pattern, normalized):
                return (
                    True,
                    reason,
                    "当前回答触发了输出安全审查，已拦截潜在的内部提示词或敏感信息。\n"
                    "如需继续咨询招生政策、流程、费用或资助问题，请换一个业务相关问题继续提问。",
                )
        return False, "ok", normalized

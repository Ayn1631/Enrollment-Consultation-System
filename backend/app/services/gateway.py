from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

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


class GatewayOrchestrator:
    WEB_SEARCH_ALLOWED_DOMAINS: tuple[str, ...] = ("zsc.zut.edu.cn", "zut.edu.cn")

    def __init__(self, deps: GatewayDependencies):
        self.deps = deps

    def create_chat(self, request: ChatRequest, fail_features: set[str] | None = None) -> ChatCreateResponse:
        """网关主流程：按 Agent 规划顺序执行功能并统一处理降级。"""
        fail_features = fail_features or set()
        # 关键变量：trace_id 用于串联网关日志、SSE 和前端故障排查。
        trace_id = uuid.uuid4().hex
        degraded: list[FeatureFlag] = []
        feature_notes: list[str] = []
        sources: list[ChatSource] = []
        context_blocks: list[str] = []
        tool_audit: list[str] = []
        status: ChatStatus = "ok"

        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "").strip()
        if not last_user:
            last_user = "请介绍中原工学院招生政策要点。"
        ordered_features = self.deps.services.plan_features(request.features)

        # 记忆读取：短期 / 长期 / 特殊分层接入。
        memory_result = self.deps.container.isolation.execute(
            "memory-service",
            lambda: self.deps.services.read_short_memory(request.session_id),
        )
        if memory_result.ok and memory_result.value and memory_result.value.entries:
            context_blocks.extend([f"[memory] {item.value}" for item in memory_result.value.entries[:3]])
            feature_notes.append("短期记忆已接入上下文。")
        else:
            degraded.append("skill_exec") if False else None
            feature_notes.append("短期记忆不可用，已忽略。")
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

        for feature in ordered_features:
            if feature == "rag":
                rag_result = self.deps.container.isolation.execute(
                    "rag-agent-service",
                    lambda: self._invoke_rag(request.session_id, last_user, fail_features),
                )
                if rag_result.ok and rag_result.value is not None:
                    rag_output = rag_result.value
                    context_blocks.extend(rag_output.context_blocks[: self.deps.services.settings.rag_final_top_k])
                    sources = [
                        ChatSource(title=item.title, url=item.url)
                        for item in rag_output.sources
                        if item.url
                    ][:5]
                    if rag_output.status == "degraded":
                        degraded.append("rag")
                        if rag_output.degrade_reason:
                            feature_notes.append(f"RAG 降级：{rag_output.degrade_reason}")
                    else:
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
                    context_blocks.extend(web_result.value)
                    feature_notes.append("联网搜索补充成功。")
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
                if guard_result.ok and guard_result.value:
                    feature_notes.append("引用校验通过。")
                else:
                    degraded.append("citation_guard")
                    feature_notes.append("引用校验失败，已启用保守模板。")

        generation_result = self.deps.container.isolation.execute(
            "generation-service",
            lambda: self._invoke_generation(
                user_query=last_user,
                context_blocks=context_blocks,
                feature_notes=feature_notes,
                request=request,
                fail_features=fail_features,
            ),
        )
        if not generation_result.ok or generation_result.value is None:
            session = SessionResult(
                session_id=request.session_id,
                trace_id=trace_id,
                text="当前生成服务异常，请稍后重试。",
                status="failed",
                degraded_features=list(dict.fromkeys(degraded)),
                sources=sources,
                tool_audit=tool_audit,
                finish_reason="error",
                error_message=generation_result.error or "generation failed",
            )
            self.deps.container.session_store.set(request.session_id, session)
            return ChatCreateResponse(
                session_id=request.session_id,
                trace_id=trace_id,
                status="failed",
                degraded_features=session.degraded_features,
            )

        generation_output = generation_result.value
        tool_audit.append(
            "generation:"
            f"{generation_output.route}:"
            f"{generation_output.model or 'unknown'}:"
            f"cache_{'hit' if generation_output.cache_hit else 'miss'}"
        )
        final_text = generation_output.text
        if "citation_guard" in request.features and (not sources or "citation_guard" in degraded):
            final_text = (
                "当前证据链不完整，以下内容仅供参考。\n"
                "建议联系招生办电话 0371-67698700 / 67698712 / 67698674 进一步确认。\n\n"
                f"{final_text}"
            )
            if "citation_guard" not in degraded:
                degraded.append("citation_guard")

        if degraded:
            status = "degraded"

        self.deps.container.isolation.execute(
            "memory-service",
            lambda: self.deps.services.write_short_memory(request.session_id, "last_user_query", last_user),
        )
        self.deps.container.isolation.execute(
            "memory-service",
            lambda: self.deps.services.append_long_memory_summary(
                request.session_id,
                self._build_long_memory_snippet(last_user=last_user, response_text=final_text),
            ),
        )
        special_preference = self._infer_special_memory(last_user)
        if special_preference is not None:
            self.deps.container.isolation.execute(
                "memory-service",
                lambda: self.deps.services.write_memory(
                    request.session_id,
                    special_preference,
                ),
            )

        session = SessionResult(
            session_id=request.session_id,
            trace_id=trace_id,
            text=final_text,
            status=status,
            degraded_features=list(dict.fromkeys(degraded)),
            sources=sources,
            tool_audit=tool_audit,
        )
        self.deps.container.session_store.set(request.session_id, session)
        return ChatCreateResponse(
            session_id=request.session_id,
            trace_id=trace_id,
            status=status,
            degraded_features=session.degraded_features,
        )

    def _invoke_rag(self, session_id: str, query: str, fail_features: set[str]):
        """执行 LangGraph RAG 调用，支持测试注入 rag 故障。"""
        if "rag" in fail_features:
            raise RuntimeError("rag failure injected")
        return self.deps.services.run_rag_graph(
            session_id=session_id,
            query=query,
            top_k=self.deps.services.settings.rag_final_top_k,
            debug=False,
        )

    def _invoke_web_search(self, query: str, fail_features: set[str]) -> list[str]:
        """执行联网搜索补充，当前为轻量占位实现。"""
        if "web_search" in fail_features:
            raise RuntimeError("web search failure injected")
        allowed = "、".join(self.WEB_SEARCH_ALLOWED_DOMAINS)
        return [f"联网补充：关于“{query}”仅允许参考 {allowed} 等白名单官方站点的最新通知。"]

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

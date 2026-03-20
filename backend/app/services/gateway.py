from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
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


class GatewayOrchestrator:
    WEB_SEARCH_ALLOWED_DOMAINS: tuple[str, ...] = ("zsc.zut.edu.cn", "zut.edu.cn")
    logger = logging.getLogger(__name__)

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
        input_blocked, input_reason, safe_reply = self._audit_user_input(last_user)
        if input_blocked:
            tool_audit.append(f"safety_audit:input_blocked:{input_reason}")
            session = SessionResult(
                session_id=request.session_id,
                trace_id=trace_id,
                text=safe_reply,
                status="degraded",
                degraded_features=[],
                sources=[],
                tool_audit=tool_audit,
                finish_reason="stop",
            )
            self.deps.container.session_store.set(request.session_id, session)
            return ChatCreateResponse(
                session_id=request.session_id,
                trace_id=trace_id,
                status="degraded",
                degraded_features=[],
            )
        route_decision = self._route_features(query=last_user, request=request)
        tool_audit.extend(route_decision.audit)
        feature_notes.extend(route_decision.notes)
        effective_features = route_decision.features
        ordered_features = self.deps.services.plan_features(effective_features)

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
            self.logger.error(
                "generation failed trace_id=%s session_id=%s error=%s features=%s",
                trace_id,
                request.session_id,
                generation_result.error or "generation failed",
                effective_features,
            )
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
        if "citation_guard" in effective_features and (not sources or "citation_guard" in degraded):
            final_text = (
                "当前证据链不完整，以下内容仅供参考。\n"
                "建议联系招生办电话 0371-67698700 / 67698712 / 67698674 进一步确认。\n\n"
                f"{final_text}"
            )
            if "citation_guard" not in degraded:
                degraded.append("citation_guard")

        output_flagged, output_reason, audited_text = self._audit_generated_output(final_text)
        if output_flagged:
            tool_audit.append(f"safety_audit:output_sanitized:{output_reason}")
            final_text = audited_text
            status = "degraded"

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

    def _route_features(self, query: str, request: ChatRequest) -> QueryRouteDecision:
        """按问题类型动态裁剪工具链，避免每个请求都一把梭全开。"""
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

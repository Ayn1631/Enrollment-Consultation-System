from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.contracts import GenerationRequest
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
        status: ChatStatus = "ok"

        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "").strip()
        if not last_user:
            last_user = "请介绍中原工学院招生政策要点。"
        ordered_features = self.deps.services.plan_features(request.features)

        # Short-term memory read (degradable)
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

        for feature in ordered_features:
            if feature == "rag":
                rag_result = self.deps.container.isolation.execute(
                    "retrieval-service",
                    lambda: self._invoke_retrieval(last_user, fail_features),
                )
                if rag_result.ok and rag_result.value is not None:
                    retrieval = rag_result.value
                    rerank_result = self.deps.container.isolation.execute(
                        "rerank-service",
                        lambda: self._invoke_rerank(last_user, retrieval, fail_features),
                    )
                    ranked = rerank_result.value if rerank_result.ok and rerank_result.value else retrieval
                    if not rerank_result.ok:
                        feature_notes.append("重排服务降级，使用原始召回结果。")
                    for item in ranked.chunks[:6]:
                        context_blocks.append(item.text)
                    sources = [ChatSource(title=item.title, url=item.url) for item in ranked.chunks if item.url][:5]
                    feature_notes.append("RAG 混合检索+重排已执行。")
                else:
                    degraded.append("rag")
                    feature_notes.append("RAG 检索失败，降级为无检索回答。")
                continue

            if feature == "web_search":
                web_result = self.deps.container.isolation.execute(
                    "web-search-service",
                    lambda: self._invoke_web_search(last_user, fail_features),
                )
                if web_result.ok and web_result.value:
                    context_blocks.extend(web_result.value)
                    feature_notes.append("联网搜索补充成功。")
                else:
                    degraded.append("web_search")
                    feature_notes.append("联网搜索失败，已降级。")
                continue

            if feature == "skill_exec":
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

        final_text = generation_result.value
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

        session = SessionResult(
            session_id=request.session_id,
            trace_id=trace_id,
            text=final_text,
            status=status,
            degraded_features=list(dict.fromkeys(degraded)),
            sources=sources,
        )
        self.deps.container.session_store.set(request.session_id, session)
        return ChatCreateResponse(
            session_id=request.session_id,
            trace_id=trace_id,
            status=status,
            degraded_features=session.degraded_features,
        )

    def _invoke_retrieval(self, query: str, fail_features: set[str]):
        if "rag" in fail_features:
            raise RuntimeError("rag failure injected")
        return self.deps.services.retrieve(query=query, top_k=10)

    def _invoke_rerank(self, query: str, retrieval, fail_features: set[str]):
        if "rerank" in fail_features:
            raise RuntimeError("rerank failure injected")
        return self.deps.services.rerank(query=query, response=retrieval, top_k=6)

    def _invoke_web_search(self, query: str, fail_features: set[str]) -> list[str]:
        if "web_search" in fail_features:
            raise RuntimeError("web search failure injected")
        return [f"联网补充：关于“{query}”请以招生官网最新通知为准。"]

    def _invoke_skill(
        self,
        query: str,
        session_id: str,
        saved_skill_id: str | None,
        fail_features: set[str],
    ) -> str:
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
    ) -> str:
        if "generation" in fail_features:
            raise RuntimeError("generation failure injected")
        result = self.deps.services.generate(
            GenerationRequest(
                user_query=user_query,
                context_blocks=context_blocks,
                feature_notes=feature_notes,
                model=request.model,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        )
        return result.text

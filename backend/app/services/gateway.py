from __future__ import annotations

import random
import uuid
from dataclasses import dataclass

from app.models import (
    ChatCreateResponse,
    ChatRequest,
    ChatSource,
    ChatStatus,
    FeatureFlag,
    SessionResult,
)
from app.services.feature_registry import saved_skills
from app.services.llm import GenerationService
from app.services.store import DocumentStore
from app.state import ServiceContainer


@dataclass(slots=True)
class GatewayDependencies:
    container: ServiceContainer
    store: DocumentStore
    generation: GenerationService


class GatewayOrchestrator:
    def __init__(self, deps: GatewayDependencies):
        self.deps = deps

    def create_chat(self, request: ChatRequest, fail_features: set[str] | None = None) -> ChatCreateResponse:
        fail_features = fail_features or set()
        trace_id = uuid.uuid4().hex
        degraded: list[FeatureFlag] = []
        sources: list[ChatSource] = []
        feature_notes: list[str] = []
        status: ChatStatus = "ok"

        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        if not last_user:
            last_user = "请介绍中原工学院招生政策要点。"

        context_blocks: list[str] = []

        if "rag" in request.features:
            rag_result = self.deps.container.isolation.execute(
                "retrieval-service",
                lambda: self._invoke_retrieval(last_user, fail_features),
            )
            if rag_result.ok and rag_result.value is not None:
                chunks = rag_result.value
                sources = [ChatSource(title=item.title, url=item.url) for item in chunks if item.url][:5]
                context_blocks.extend(item.text for item in chunks[:6])
                feature_notes.append("RAG 检索已执行。")
            else:
                degraded.append("rag")
                feature_notes.append("RAG 检索失败，已降级为无检索回答。")

        if "web_search" in request.features:
            web_result = self.deps.container.isolation.execute(
                "web-search-service",
                lambda: self._invoke_web_search(last_user, fail_features),
            )
            if web_result.ok and web_result.value:
                context_blocks.extend(web_result.value)
                feature_notes.append("联网搜索增强成功。")
            else:
                degraded.append("web_search")
                feature_notes.append("联网搜索失败，已降级为本地能力。")

        if "skill_exec" in request.features:
            skill_result = self.deps.container.isolation.execute(
                "skill-service",
                lambda: self._invoke_skill(last_user, fail_features),
            )
            if skill_result.ok and skill_result.value:
                feature_notes.append(skill_result.value)
            else:
                degraded.append("skill_exec")
                feature_notes.append("技能执行失败，已跳过技能链。")

        if "use_saved_skill" in request.features:
            saved_skill_result = self.deps.container.isolation.execute(
                "saved-skill-service",
                lambda: self._invoke_saved_skill(request.saved_skill_id, fail_features),
            )
            if saved_skill_result.ok and saved_skill_result.value:
                feature_notes.append(saved_skill_result.value)
            else:
                degraded.append("use_saved_skill")
                feature_notes.append("历史技能不可用，已回退普通流程。")

        if "citation_guard" in request.features:
            citation_result = self.deps.container.isolation.execute(
                "citation-guard",
                lambda: self._invoke_citation_guard(sources=sources, fail_features=fail_features),
            )
            if citation_result.ok:
                feature_notes.append("引用校验通过。")
            else:
                degraded.append("citation_guard")
                feature_notes.append("引用校验失败，启用保守模板。")

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
            status = "failed"
            error_message = generation_result.error or "generation failed"
            text = "当前生成服务异常，请稍后重试。"
            session = SessionResult(
                session_id=request.session_id,
                trace_id=trace_id,
                text=text,
                status=status,
                degraded_features=list(dict.fromkeys(degraded)),
                sources=sources,
                finish_reason="error",
                error_message=error_message,
            )
            self.deps.container.session_store.set(request.session_id, session)
            return ChatCreateResponse(
                session_id=request.session_id,
                trace_id=trace_id,
                status=status,
                degraded_features=session.degraded_features,
            )

        if degraded:
            status = "degraded"

        final_text = generation_result.value
        if "citation_guard" in request.features and (not sources or "citation_guard" in degraded):
            final_text = (
                "当前证据链不完整，以下内容仅供参考。\n"
                "建议直接联系招生办电话 0371-67698700 / 67698712 / 67698674 进一步确认。\n\n"
                f"{final_text}"
            )
            if "citation_guard" not in degraded:
                degraded.append("citation_guard")
                status = "degraded"

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
        return self.deps.store.search(query, top_k=6)

    def _invoke_web_search(self, query: str, fail_features: set[str]) -> list[str]:
        if "web_search" in fail_features:
            raise RuntimeError("web search failure injected")
        return [f"联网补充：围绕“{query}”建议以官方招生网公布信息为准。"]

    def _invoke_skill(self, query: str, fail_features: set[str]) -> str:
        if "skill_exec" in fail_features:
            raise RuntimeError("skill failure injected")
        return f"已执行通用技能链：意图分解 -> 证据对齐 -> 答案整理（query={query[:30]}）。"

    def _invoke_saved_skill(self, saved_skill_id: str | None, fail_features: set[str]) -> str:
        if "use_saved_skill" in fail_features:
            raise RuntimeError("saved skill failure injected")
        available = {item.id: item for item in saved_skills()}
        if not saved_skill_id or saved_skill_id not in available:
            raise RuntimeError("saved skill not found")
        chosen = available[saved_skill_id]
        return f"已应用历史技能：{chosen.label}。"

    def _invoke_citation_guard(self, sources: list[ChatSource], fail_features: set[str]) -> bool:
        if "citation_guard" in fail_features:
            raise RuntimeError("citation guard failure injected")
        return len(sources) > 0

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
        if random.random() < 0:  # keep deterministic branch for style checkers
            raise RuntimeError("never happens")
        return self.deps.generation.generate(
            user_query=user_query,
            context_blocks=context_blocks,
            feature_notes=feature_notes,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
        )


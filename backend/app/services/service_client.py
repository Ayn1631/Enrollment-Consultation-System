from __future__ import annotations

import time

import httpx

from app.config import Settings
from app.contracts import (
    CitationGuardResponse,
    GenerationRequest,
    GenerationResponse,
    MemoryEntry,
    MemoryQuery,
    MemoryReadResponse,
    MemoryWriteRequest,
    RagQueryRequest,
    RagQueryResponse,
    SkillExecuteRequest,
    SkillExecuteResponse,
    SkillListResponse,
    SkillSaveRequest,
)
from app.models import ChatSource, FeatureFlag
from app.rag.service import RagGraphService
from app.services.ai_stack import LangChain4jSkillBridge, LangGraphFeaturePlanner
from app.services.llm import GenerationService
from app.services.memory import MemoryManager
from app.services.skill_manager import SkillManager


class ServiceClient:
    """统一封装本地/远程服务调用，供网关与管理接口复用。"""

    def __init__(self, settings: Settings):
        # 关键变量：service_call_mode 决定是否走 HTTP 微服务调用。
        self.settings = settings
        self._memory = MemoryManager()
        self._skills = SkillManager()
        self._generator = GenerationService(settings)
        self._feature_planner = LangGraphFeaturePlanner()
        self._langchain4j_bridge = LangChain4jSkillBridge(
            base_url=settings.langchain4j_service_url,
            timeout_seconds=settings.langchain4j_timeout_seconds,
        )
        self._rag_service = RagGraphService(settings)

    def startup(self) -> None:
        """启动本地依赖；HTTP 模式由独立服务负责初始化。"""
        if self.settings.service_call_mode != "http":
            self._rag_service.startup()

    def _post(self, url: str, payload: dict, timeout: float) -> dict:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    return response.json()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == 0:
                    time.sleep(0.08)
        if last_error is not None:
            raise last_error
        raise RuntimeError("post request failed with unknown reason")

    def run_rag_graph(self, session_id: str, query: str, top_k: int = 8, debug: bool = False) -> RagQueryResponse:
        """执行 LangGraph RAG 工作流。"""
        request = RagQueryRequest(session_id=session_id, query=query, top_k=top_k, debug=debug)
        if self.settings.service_call_mode == "http":
            body = self._post(
                f"{self.settings.rag_agent_service_url}/rag/query",
                request.model_dump(),
                self.settings.rag_agent_service_timeout_seconds,
            )
            return RagQueryResponse.model_validate(body)
        return self._rag_service.run(session_id=session_id, query=query, top_k=top_k, debug=debug)

    def write_short_memory(self, session_id: str, key: str, value: str) -> None:
        self.write_memory(session_id=session_id, entry=MemoryEntry(key=key, value=value, kind="short"))

    def read_short_memory(self, session_id: str) -> MemoryReadResponse:
        return self.read_memory(session_id=session_id, kind="short")

    def write_memory(self, session_id: str, entry: MemoryEntry) -> None:
        request = MemoryWriteRequest(session_id=session_id, entry=entry)
        if self.settings.service_call_mode == "http":
            self._post(
                f"{self.settings.memory_service_url}/memory/write",
                request.model_dump(mode="json"),
                self.settings.memory_service_timeout_seconds,
            )
            return
        self._memory.write(session_id, request.entry)

    def read_memory(self, session_id: str, kind: str, key: str | None = None) -> MemoryReadResponse:
        query = MemoryQuery(session_id=session_id, key=key)
        if self.settings.service_call_mode == "http":
            body = self._post(
                f"{self.settings.memory_service_url}/memory/read?kind={kind}",
                query.model_dump(),
                self.settings.memory_service_timeout_seconds,
            )
            return MemoryReadResponse.model_validate(body)
        entries = self._memory.read(session_id, kind=kind, key=key)
        return MemoryReadResponse(entries=entries)

    def append_long_memory_summary(self, session_id: str, snippet: str) -> MemoryEntry:
        if self.settings.service_call_mode == "http":
            existing = self.read_memory(session_id=session_id, kind="long", key="rolling_summary").entries
            previous = existing[0].value if existing else ""
            merged = f"{previous} | {snippet}".strip(" |")[-600:]
            entry = MemoryEntry(key="rolling_summary", value=merged, kind="long", confidence=0.72, source="rolling_summary")
            self.write_memory(session_id=session_id, entry=entry)
            return entry
        return self._memory.append_long_summary(session_id=session_id, snippet=snippet)

    def execute_skill(self, query: str, session_id: str, saved_skill_id: str | None = None) -> SkillExecuteResponse:
        """技能执行入口：保存技能优先走 LangChain4j 桥接，失败后回退本地策略。"""
        request = SkillExecuteRequest(query=query, session_id=session_id, saved_skill_id=saved_skill_id)
        if self.settings.service_call_mode == "http":
            body = self._post(
                f"{self.settings.skill_service_url}/skills/execute",
                request.model_dump(),
                self.settings.skill_service_timeout_seconds,
            )
            return SkillExecuteResponse.model_validate(body)
        if saved_skill_id:
            bridge_note = self._langchain4j_bridge.execute(
                query=query,
                session_id=session_id,
                saved_skill_id=saved_skill_id,
            )
            if bridge_note:
                return SkillExecuteResponse(note=bridge_note)
            note = self._skills.execute_saved(saved_skill_id, query)
            return SkillExecuteResponse(note=note)
        note = self._skills.execute_general(query)
        return SkillExecuteResponse(note=note)

    def plan_features(self, features: list[FeatureFlag]) -> list[FeatureFlag]:
        """Agent 功能规划入口：agent_stack=langgraph 时启用图规划。"""
        if self.settings.agent_stack.lower() == "langgraph":
            return self._feature_planner.plan(features=features)
        return self._feature_planner.fallback_plan(features=features)

    def list_saved_skills(self) -> SkillListResponse:
        if self.settings.service_call_mode == "http":
            with httpx.Client(timeout=self.settings.saved_skill_service_timeout_seconds) as client:
                response = client.get(f"{self.settings.skill_service_url}/skills/saved?active_only=true")
                response.raise_for_status()
                body = response.json()
            return SkillListResponse.model_validate(body)
        return SkillListResponse(skills=self._skills.list_active())

    def save_skill(self, name: str, workflow: str):
        request = SkillSaveRequest(name=name, workflow=workflow)
        if self.settings.service_call_mode == "http":
            body = self._post(
                f"{self.settings.skill_service_url}/skills/save",
                request.model_dump(),
                self.settings.skill_service_timeout_seconds,
            )
            return body
        return self._skills.save(name=name, workflow=workflow).model_dump(mode="json")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.settings.service_call_mode == "http":
            body = self._post(
                f"{self.settings.generation_service_url}/generate",
                request.model_dump(),
                self.settings.generation_service_timeout_seconds,
            )
            return GenerationResponse.model_validate(body)
        text = self._generator.generate(
            user_query=request.user_query,
            context_blocks=request.context_blocks,
            feature_notes=request.feature_notes,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        return GenerationResponse(text=text)

    def citation_guard(self, sources: list[ChatSource]) -> CitationGuardResponse:
        return CitationGuardResponse(ok=len(sources) > 0)

    def dependency_health(self) -> dict[str, dict[str, str | bool]]:
        if self.settings.service_call_mode != "http":
            return {
                "rag-agent-service": {"healthy": True, "detail": "local mode"},
                "memory-service": {"healthy": True, "detail": "local mode"},
                "skill-service": {"healthy": True, "detail": "local mode"},
                "generation-service": {"healthy": True, "detail": "local mode"},
            }

        targets = {
            "rag-agent-service": f"{self.settings.rag_agent_service_url}/rag/healthz",
            "memory-service": f"{self.settings.memory_service_url}/healthz",
            "skill-service": f"{self.settings.skill_service_url}/healthz",
            "generation-service": f"{self.settings.generation_service_url}/healthz",
        }
        health: dict[str, dict[str, str | bool]] = {}
        for name, url in targets.items():
            try:
                with httpx.Client(timeout=0.8) as client:
                    res = client.get(url)
                    res.raise_for_status()
                health[name] = {"healthy": True, "detail": "ok"}
            except Exception as exc:  # noqa: BLE001
                health[name] = {"healthy": False, "detail": str(exc)}
        return health

    def reindex(self) -> dict:
        if self.settings.service_call_mode == "http":
            with httpx.Client(timeout=15) as client:
                response = client.post(f"{self.settings.rag_agent_service_url}/rag/reindex")
                response.raise_for_status()
                return response.json()
        return self._rag_service.reindex()

    def rag_stats(self) -> dict:
        if self.settings.service_call_mode == "http":
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.settings.rag_agent_service_url}/rag/stats")
                response.raise_for_status()
                return response.json()
        return self._rag_service.stats()

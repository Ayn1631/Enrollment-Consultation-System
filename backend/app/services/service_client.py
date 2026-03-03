from __future__ import annotations

import httpx
import time

from app.config import Settings
from app.contracts import (
    CitationGuardResponse,
    GenerationRequest,
    GenerationResponse,
    MemoryEntry,
    MemoryQuery,
    MemoryReadResponse,
    MemoryWriteRequest,
    RerankRequest,
    RerankResponse,
    RetrievalRequest,
    RetrievalResponse,
    SkillExecuteRequest,
    SkillExecuteResponse,
    SkillListResponse,
    SkillSaveRequest,
)
from app.models import ChatSource
from app.services.llm import GenerationService
from app.services.memory import MemoryManager
from app.services.reranker import SimpleReranker
from app.services.skill_manager import SkillManager
from app.services.store import DocumentStore


class ServiceClient:
    def __init__(self, settings: Settings, store: DocumentStore):
        self.settings = settings
        self._store = store
        self._reranker = SimpleReranker()
        self._memory = MemoryManager()
        self._skills = SkillManager()
        self._generator = GenerationService(settings)

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

    def retrieve(self, query: str, top_k: int = 8) -> RetrievalResponse:
        if self.settings.service_call_mode == "http":
            body = self._post(
                f"{self.settings.retrieval_service_url}/retrieve",
                RetrievalRequest(query=query, top_k=top_k).model_dump(),
                self.settings.retrieval_service_timeout_seconds,
            )
            return RetrievalResponse.model_validate(body)
        chunks = self._store.search(query, top_k=top_k)
        return RetrievalResponse(
            chunks=[
                {
                    "chunk_id": item.chunk_id,
                    "title": item.title,
                    "url": item.url,
                    "text": item.text,
                    "score": item.score,
                    "bm25_score": item.bm25_score,
                    "vector_score": item.vector_score,
                    "keyword_score": item.keyword_score,
                }
                for item in chunks
            ]
        )

    def rerank(self, query: str, response: RetrievalResponse, top_k: int = 6) -> RerankResponse:
        request = RerankRequest(query=query, chunks=response.chunks, top_k=top_k)
        if self.settings.service_call_mode == "http":
            body = self._post(
                f"{self.settings.rerank_service_url}/rerank",
                request.model_dump(),
                self.settings.rerank_service_timeout_seconds,
            )
            return RerankResponse.model_validate(body)
        ranked = self._reranker.rerank(query, response.chunks, top_k=top_k)
        return RerankResponse(chunks=ranked)

    def write_short_memory(self, session_id: str, key: str, value: str) -> None:
        request = MemoryWriteRequest(
            session_id=session_id,
            entry=MemoryEntry(key=key, value=value, kind="short"),
        )
        if self.settings.service_call_mode == "http":
            self._post(
                f"{self.settings.memory_service_url}/memory/write",
                request.model_dump(mode="json"),
                self.settings.memory_service_timeout_seconds,
            )
            return
        self._memory.write(session_id, request.entry)

    def read_short_memory(self, session_id: str) -> MemoryReadResponse:
        query = MemoryQuery(session_id=session_id)
        if self.settings.service_call_mode == "http":
            body = self._post(
                f"{self.settings.memory_service_url}/memory/read?kind=short",
                query.model_dump(),
                self.settings.memory_service_timeout_seconds,
            )
            return MemoryReadResponse.model_validate(body)
        entries = self._memory.read(session_id, kind="short")
        return MemoryReadResponse(entries=entries)

    def execute_skill(self, query: str, session_id: str, saved_skill_id: str | None = None) -> SkillExecuteResponse:
        request = SkillExecuteRequest(query=query, session_id=session_id, saved_skill_id=saved_skill_id)
        if self.settings.service_call_mode == "http":
            body = self._post(
                f"{self.settings.skill_service_url}/skills/execute",
                request.model_dump(),
                self.settings.skill_service_timeout_seconds,
            )
            return SkillExecuteResponse.model_validate(body)
        if saved_skill_id:
            note = self._skills.execute_saved(saved_skill_id, query)
            return SkillExecuteResponse(note=note)
        note = self._skills.execute_general(query)
        return SkillExecuteResponse(note=note)

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
        # Citation guard is lightweight and always local.
        return CitationGuardResponse(ok=len(sources) > 0)

    def dependency_health(self) -> dict[str, dict[str, str | bool]]:
        if self.settings.service_call_mode != "http":
            return {
                "retrieval-service": {"healthy": True, "detail": "local mode"},
                "rerank-service": {"healthy": True, "detail": "local mode"},
                "memory-service": {"healthy": True, "detail": "local mode"},
                "skill-service": {"healthy": True, "detail": "local mode"},
                "generation-service": {"healthy": True, "detail": "local mode"},
            }

        targets = {
            "retrieval-service": f"{self.settings.retrieval_service_url}/healthz",
            "rerank-service": f"{self.settings.rerank_service_url}/healthz",
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
            with httpx.Client(timeout=10) as client:
                response = client.post(f"{self.settings.retrieval_service_url}/reindex")
                response.raise_for_status()
                return response.json()
        self._store.load()
        return {"chunks": len(self._store.chunks)}

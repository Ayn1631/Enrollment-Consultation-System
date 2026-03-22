from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models import ChatSource


class RagEvidence(BaseModel):
    chunk_id: str
    title: str
    url: str
    text: str
    score: float


class RagQueryRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 8
    debug: bool = False
    memory_context_blocks: list[str] = Field(default_factory=list)


class RagQueryResponse(BaseModel):
    trace_id: str
    status: Literal["ok", "degraded"] = "ok"
    context_blocks: list[str] = Field(default_factory=list)
    sources: list[RagEvidence] = Field(default_factory=list)
    degrade_reason: str | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)


class GenerationRequest(BaseModel):
    user_query: str
    context_blocks: list[str] = Field(default_factory=list)
    feature_notes: list[str] = Field(default_factory=list)
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None


GenerationRoute = Literal["light", "main", "requested", "mock"]


class GenerationResponse(BaseModel):
    text: str
    model: str = ""
    route: GenerationRoute = "requested"
    cache_hit: bool = False


class GenerationStreamChunk(BaseModel):
    delta: str = ""
    done: bool = False
    response: GenerationResponse | None = None


class MemoryEntry(BaseModel):
    key: str
    value: str
    kind: str
    confidence: float = 0.9
    source: str = "system"
    last_verified: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None


class MemoryQuery(BaseModel):
    session_id: str
    key: str | None = None


class MemoryWriteRequest(BaseModel):
    session_id: str
    entry: MemoryEntry


class MemoryReadResponse(BaseModel):
    entries: list[MemoryEntry] = Field(default_factory=list)


class SkillExecuteRequest(BaseModel):
    query: str
    session_id: str
    saved_skill_id: str | None = None


class SkillExecuteResponse(BaseModel):
    note: str


class SkillSaveRequest(BaseModel):
    name: str
    workflow: str


class SkillVersionItem(BaseModel):
    id: str
    name: str
    version: int
    workflow_hash: str
    score: float
    active: bool
    created_at: datetime
    description: str


class SkillListResponse(BaseModel):
    skills: list[SkillVersionItem] = Field(default_factory=list)


class CitationGuardRequest(BaseModel):
    sources: list[ChatSource] = Field(default_factory=list)


class CitationGuardResponse(BaseModel):
    ok: bool


@dataclass(slots=True)
class GatewayFeatureContext:
    context_blocks: list[str]
    sources: list[ChatSource]
    notes: list[str]

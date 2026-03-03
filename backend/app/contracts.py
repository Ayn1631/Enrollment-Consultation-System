from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field

from app.models import ChatSource


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 8


class RetrievalChunk(BaseModel):
    chunk_id: str
    title: str
    url: str
    text: str
    score: float
    bm25_score: float
    vector_score: float
    keyword_score: float


class RetrievalResponse(BaseModel):
    chunks: list[RetrievalChunk] = Field(default_factory=list)


class RerankRequest(BaseModel):
    query: str
    chunks: list[RetrievalChunk] = Field(default_factory=list)
    top_k: int = 6


class RerankResponse(BaseModel):
    chunks: list[RetrievalChunk] = Field(default_factory=list)


class GenerationRequest(BaseModel):
    user_query: str
    context_blocks: list[str] = Field(default_factory=list)
    feature_notes: list[str] = Field(default_factory=list)
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None


class GenerationResponse(BaseModel):
    text: str


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


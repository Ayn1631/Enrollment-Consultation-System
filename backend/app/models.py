from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


Role = Literal["user", "assistant", "system"]
ChatMode = Literal["chat", "plan", "guide"]

FeatureFlag = Literal["rag", "web_search", "skill_exec", "use_saved_skill", "citation_guard"]
LegacyToolMode = Literal["search", "react", "plan", "guide"]

ChatStatus = Literal["ok", "degraded", "failed"]
FinishReason = Literal["stop", "length", "error"]


DEFAULT_FEATURES: tuple[FeatureFlag, ...] = ("rag", "citation_guard")

LEGACY_TO_FEATURE: dict[LegacyToolMode, FeatureFlag] = {
    "search": "web_search",
    "react": "skill_exec",
    "plan": "skill_exec",
    "guide": "skill_exec",
}

FEATURE_DEPENDENCIES: dict[FeatureFlag, tuple[FeatureFlag, ...]] = {
    "use_saved_skill": ("skill_exec",),
    "citation_guard": ("rag",),
}


def _expand_feature_dependencies(features: list[FeatureFlag]) -> list[FeatureFlag]:
    """按依赖关系补齐特性集合，保持输入顺序并避免重复。"""
    # 关键变量：expanded 是最终输出的有序特性列表。
    expanded = list(dict.fromkeys(features))
    changed = True
    while changed:
        changed = False
        for feature in list(expanded):
            for dependency in FEATURE_DEPENDENCIES.get(feature, ()):
                if dependency not in expanded:
                    expanded.append(dependency)
                    changed = True
    return expanded


class ChatMessageInput(BaseModel):
    role: Role
    content: str


class ChatSource(BaseModel):
    title: str
    url: str


class ChatRequest(BaseModel):
    session_id: str
    messages: list[ChatMessageInput] = Field(default_factory=list)
    mode: ChatMode = "chat"
    stream: bool = True
    features: list[FeatureFlag] = Field(default_factory=list)
    tools: list[LegacyToolMode] = Field(default_factory=list)
    saved_skill_id: str | None = None
    strict_citation: bool = False
    temperature: float | None = None
    top_p: float | None = None
    model: str | None = None

    @field_validator("features")
    @classmethod
    def unique_features(cls, value: list[FeatureFlag]) -> list[FeatureFlag]:
        deduped = list(dict.fromkeys(value))
        return deduped

    @model_validator(mode="after")
    def normalize_features(self) -> "ChatRequest":
        """归一化功能开关：兼容 legacy tools、默认值与依赖补全。"""
        normalized = list(self.features)
        for item in self.tools:
            mapped = LEGACY_TO_FEATURE.get(item)
            if mapped and mapped not in normalized:
                normalized.append(mapped)
        if not normalized:
            normalized = list(DEFAULT_FEATURES)
        if self.strict_citation and "citation_guard" not in normalized:
            normalized.append("citation_guard")
        normalized = _expand_feature_dependencies(normalized)
        self.features = normalized
        if "use_saved_skill" in self.features and not self.saved_skill_id:
            raise ValueError("saved_skill_id is required when use_saved_skill feature is enabled.")
        return self


class ChatCreateResponse(BaseModel):
    session_id: str
    trace_id: str
    status: ChatStatus
    degraded_features: list[FeatureFlag] = Field(default_factory=list)


class ChatStreamDone(BaseModel):
    finish_reason: FinishReason = "stop"
    status: ChatStatus = "ok"
    degraded_features: list[FeatureFlag] = Field(default_factory=list)
    sources: list[ChatSource] = Field(default_factory=list)
    trace_id: str = ""
    tool_audit: list[str] = Field(default_factory=list)


class FeatureMeta(BaseModel):
    id: FeatureFlag
    label: str
    default_enabled: bool = False
    dependencies: list[FeatureFlag] = Field(default_factory=list)


class SavedSkill(BaseModel):
    id: str
    label: str
    description: str


class HealthDependency(BaseModel):
    name: str
    healthy: bool
    circuit_open: bool
    last_error: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    app: str
    healthy: bool
    status: ChatStatus | None = None
    dependencies: list[HealthDependency] = Field(default_factory=list)


class SessionResult(BaseModel):
    session_id: str
    trace_id: str
    text: str
    status: ChatStatus
    degraded_features: list[FeatureFlag] = Field(default_factory=list)
    sources: list[ChatSource] = Field(default_factory=list)
    tool_audit: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finish_reason: FinishReason = "stop"
    error_message: str | None = None

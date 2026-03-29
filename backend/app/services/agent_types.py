from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from app.models import AgentStepEvent, AgentStrategy, ChatRequest, ChatSource, FeatureFlag


PlanStepType = Literal[
    "recall_memory",
    "local_rag_search",
    "general_skill",
    "saved_skill",
    "mcp_discover",
    "mcp_execute",
    "citation_guard",
    "synthesize_step",
]


SubproblemStatus = Literal["pending", "completed", "degraded", "failed", "needs_replan"]


@dataclass(slots=True)
class PlanStep:
    step_type: PlanStepType
    title: str
    instruction: str = ""


@dataclass(slots=True)
class SubproblemState:
    subproblem_id: str
    query: str
    plan_steps: list[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    attempt_count: int = 0
    step_outputs: dict[str, str] = field(default_factory=dict)
    status: SubproblemStatus = "pending"
    degraded: bool = False
    notes: list[str] = field(default_factory=list)
    context_blocks: list[str] = field(default_factory=list)
    sources: list[ChatSource] = field(default_factory=list)
    tool_audit: list[str] = field(default_factory=list)
    replan_count: int = 0


@dataclass(slots=True)
class StepExecutionResult:
    ok: bool
    message: str = ""
    context_blocks: list[str] = field(default_factory=list)
    sources: list[ChatSource] = field(default_factory=list)
    tool_audit: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StepReviewResult:
    ok: bool
    message: str


class AgentGraphState(TypedDict):
    trace_id: str
    session_id: str
    last_user: str
    request: ChatRequest
    fail_features: set[str]
    agent_strategy: AgentStrategy
    effective_features: list[FeatureFlag]
    route_label: str
    route_reason: str
    memory_context: list[str]
    memory_text: str
    rewritten_query: str
    subproblems: list[SubproblemState]
    current_subproblems: list[SubproblemState]
    subproblem_results: list[SubproblemState]
    final_text: str
    sources: list[ChatSource]
    tool_audit: list[str]
    notes: list[str]
    degraded_features: list[FeatureFlag]
    step_events: list[AgentStepEvent]
    failure_reason: str | None
    pending_retries: list[SubproblemState]
    generation_context_blocks: list[str]
    generation_notes: list[str]
    merge_summary: str
    blocked_reply: str | None
    blocked_audit: list[str]
    current_round_complete: bool
    status: Literal["ok", "degraded", "failed"]

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langchain_core.documents import Document


@dataclass(slots=True)
class RagRunOptions:
    """RAG 运行参数，供网关按会话级策略覆盖默认值。"""

    top_k: int = 8
    debug: bool = False


class RagGraphState(TypedDict, total=False):
    """LangGraph 工作流共享状态。"""

    trace_id: str
    session_id: str
    raw_query: str
    normalized_query: str
    route_label: str
    route_reason: str
    route_retrieve_top_n: int
    summary_focus_parent_ids: list[str]
    rewritten_queries: list[str]
    retrieved_docs: list[Document]
    reranked_docs: list[Document]
    final_context_blocks: list[str]
    quality_passed: bool
    quality_report: dict[str, float | int | bool | str | None]
    retry_count: int
    guard_passed: bool
    degrade_reason: str | None
    latency_breakdown_ms: dict[str, float]

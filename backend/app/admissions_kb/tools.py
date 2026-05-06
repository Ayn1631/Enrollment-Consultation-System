from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.admissions_kb.repository import AdmissionsRepository
from app.contracts import RagEvidence, RagQueryResponse
from app.models import ChatSource


@dataclass(slots=True)
class StructuredToolPayload:
    tool_name: str
    records: list[dict]
    matched_fields: list[str]
    route_reason: str


class StructuredAdmissionsToolset:
    def __init__(self, settings):
        self.settings = settings
        self.repository = AdmissionsRepository(settings)

    def major_catalog_lookup(self, *, raw_query: str, filters: dict[str, str], limit: int = 8) -> StructuredToolPayload:
        records = self.repository.search_major_catalog(raw_query=raw_query, filters=filters, limit=limit)
        return StructuredToolPayload(
            tool_name="major_catalog_lookup",
            records=records,
            matched_fields=[key for key, value in filters.items() if value],
            route_reason="专业目录类结构化查询",
        )

    def scoreline_lookup(self, *, raw_query: str, filters: dict[str, str], limit: int = 8) -> StructuredToolPayload:
        records = self.repository.search_score_lines(raw_query=raw_query, filters=filters, limit=limit)
        return StructuredToolPayload(
            tool_name="scoreline_lookup",
            records=records,
            matched_fields=[key for key, value in filters.items() if value],
            route_reason="录取分数/位次类结构化查询",
        )

    def policy_table_lookup(self, *, raw_query: str, filters: dict[str, str], limit: int = 12) -> StructuredToolPayload:
        records = self.repository.search_policy_tables(raw_query=raw_query, filters=filters, limit=limit)
        return StructuredToolPayload(
            tool_name="policy_table_lookup",
            records=records,
            matched_fields=[key for key, value in filters.items() if value],
            route_reason="政策附表类结构化查询",
        )

    def to_rag_response(self, *, payload: StructuredToolPayload, trace_id: str) -> RagQueryResponse | None:
        if not payload.records:
            return None
        context_blocks: list[str] = []
        sources: list[RagEvidence] = []
        for index, record in enumerate(payload.records, start=1):
            evidence_text = str(record.get("evidence_text", "")).strip()
            source_file = str(record.get("source_file", "")).strip()
            source_url = self._resolve_source_path(source_file)
            title = self._resolve_title(payload.tool_name, record)
            context_blocks.append(
                f"[structured:{payload.tool_name}][matched={','.join(payload.matched_fields) or 'raw_query'}]\n"
                f"标题：{title}\n"
                f"来源文件：{source_file}\n"
                f"证据：{evidence_text}"
            )
            sources.append(
                RagEvidence(
                    chunk_id=f"{payload.tool_name}:{index}",
                    title=title,
                    url=source_url,
                    text=evidence_text,
                    score=max(0.9 - index * 0.05, 0.5),
                )
            )
        return RagQueryResponse(
            trace_id=trace_id,
            status="ok",
            context_blocks=context_blocks,
            sources=sources,
            degrade_reason=None,
            latency_ms={"structured_route": 0.0},
        )

    def to_chat_sources(self, payload: StructuredToolPayload) -> list[ChatSource]:
        sources: list[ChatSource] = []
        for record in payload.records:
            source_file = str(record.get("source_file", "")).strip()
            sources.append(ChatSource(title=self._resolve_title(payload.tool_name, record), url=self._resolve_source_path(source_file)))
        return sources

    def _resolve_title(self, tool_name: str, record: dict) -> str:
        if tool_name == "major_catalog_lookup":
            return f"{record.get('major_name', '')} - {record.get('college_name', '')}".strip(" -")
        if tool_name == "scoreline_lookup":
            return f"{record.get('year', '')} {record.get('province', '')} {record.get('major_name', '')}".strip()
        return str(record.get("table_topic", "")) or str(record.get("source_doc", "政策附表"))

    def _resolve_source_path(self, source_file: str) -> str:
        if not source_file:
            return ""
        candidates = [
            Path(self.settings.admissions_source_dir) / source_file,
            Path(self.settings.admissions_source_dir) / "衍生数据" / source_file,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return source_file

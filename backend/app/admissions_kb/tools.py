from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from app.admissions_kb.parsers import load_major_catalog_rows, load_score_line_rows
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
        self._major_cache: list[dict] | None = None
        self._score_cache: list[dict] | None = None
        self._policy_cache: list[dict] | None = None

    def major_catalog_lookup(self, *, raw_query: str, filters: dict[str, str], limit: int = 8) -> StructuredToolPayload:
        try:
            records = self.repository.search_major_catalog(raw_query=raw_query, filters=filters, limit=limit)
        except Exception:
            records = self._search_major_catalog_fallback(raw_query=raw_query, filters=filters, limit=limit)
        if not records:
            records = self._search_major_catalog_fallback(raw_query=raw_query, filters=filters, limit=limit)
        return StructuredToolPayload(
            tool_name="major_catalog_lookup",
            records=records,
            matched_fields=[key for key, value in filters.items() if value],
            route_reason="专业目录类结构化查询",
        )

    def scoreline_lookup(self, *, raw_query: str, filters: dict[str, str], limit: int = 8) -> StructuredToolPayload:
        try:
            records = self.repository.search_score_lines(raw_query=raw_query, filters=filters, limit=limit)
        except Exception:
            records = self._search_scoreline_fallback(raw_query=raw_query, filters=filters, limit=limit)
        if not records:
            records = self._search_scoreline_fallback(raw_query=raw_query, filters=filters, limit=limit)
        return StructuredToolPayload(
            tool_name="scoreline_lookup",
            records=records,
            matched_fields=[key for key, value in filters.items() if value],
            route_reason="录取分数/位次类结构化查询",
        )

    def policy_table_lookup(self, *, raw_query: str, filters: dict[str, str], limit: int = 12) -> StructuredToolPayload:
        try:
            records = self.repository.search_policy_tables(raw_query=raw_query, filters=filters, limit=limit)
        except Exception:
            records = self._search_policy_fallback(raw_query=raw_query, filters=filters, limit=limit)
        if not records:
            records = self._search_policy_fallback(raw_query=raw_query, filters=filters, limit=limit)
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

    def _search_major_catalog_fallback(self, *, raw_query: str, filters: dict[str, str], limit: int) -> list[dict]:
        if self._major_cache is None:
            source_root = Path(self.settings.admissions_source_dir)
            derived = source_root / "衍生数据" / "中原工学院2025年本科招生章程-结构化附表.xlsx"
            rows = load_major_catalog_rows(source_root / "2025年招生专业详情.xlsx", source_dataset="major_catalog")
            if derived.exists():
                rows.extend(load_major_catalog_rows(derived, source_dataset="major_catalog"))
            self._major_cache = rows
        return self._filter_rows(self._major_cache, raw_query=raw_query, filters=filters, limit=limit)

    def _search_scoreline_fallback(self, *, raw_query: str, filters: dict[str, str], limit: int) -> list[dict]:
        if self._score_cache is None:
            try:
                self._score_cache = load_score_line_rows(Path(self.settings.admissions_source_dir) / "2025年录取分数线.xls")
            except Exception:
                self._score_cache = []
        return self._filter_rows(self._score_cache, raw_query=raw_query, filters=filters, limit=limit)

    def _search_policy_fallback(self, *, raw_query: str, filters: dict[str, str], limit: int) -> list[dict]:
        if self._policy_cache is None:
            self._policy_cache = []
            for row in self._search_major_catalog_fallback(raw_query="", filters={}, limit=400):
                if not row.get("source_table_title"):
                    continue
                self._policy_cache.append(
                    {
                        "source_dataset": "policy_tables",
                        "source_file": row.get("source_file", ""),
                        "source_doc": row.get("source_doc", ""),
                        "table_topic": row.get("source_table_title", ""),
                        "source_row_no": row.get("source_row_no", ""),
                        "field_name": "evidence_text",
                        "field_value": row.get("evidence_text", ""),
                        "evidence_text": row.get("evidence_text", ""),
                    }
                )
        return self._filter_rows(self._policy_cache, raw_query=raw_query, filters=filters, limit=limit)

    def _filter_rows(self, rows: list[dict], *, raw_query: str, filters: dict[str, str], limit: int) -> list[dict]:
        matched: list[tuple[int, dict]] = []
        query_text = raw_query.strip()
        normalized_query = self._normalize_text(query_text)
        for row in rows:
            score = 0
            row_text = " ".join(str(value) for value in row.values())
            normalized_row = self._normalize_text(row_text)
            if normalized_query and normalized_query in normalized_row:
                score += 6
            if not self._matches_numeric_filters(row=row, filters=filters):
                continue
            score += self._score_filter_matches(row=row, filters=filters, normalized_row=normalized_row)
            if score <= 0 and normalized_query:
                score += sum(1 for token in self._tokenize_query(query_text) if token in normalized_row)
            if score <= 0:
                continue
            matched.append((score, row))
        matched.sort(
            key=lambda item: (
                -item[0],
                self._safe_int(item[1].get("source_row_no", "")) or 0,
            ),
        )
        return [row for _, row in matched[:limit]]

    def _score_filter_matches(self, *, row: dict, filters: dict[str, str], normalized_row: str) -> int:
        score = 0
        hard_fields = {
            "academic_year",
            "year",
            "major_code",
            "major_name",
            "college_name",
            "province",
            "batch",
            "category",
            "exam_subjects",
            "degree_type",
            "table_topic",
            "keyword",
        }
        for key, value in filters.items():
            if not value or key.endswith("_min") or key.endswith("_max"):
                continue
            normalized_value = self._normalize_text(str(value))
            if not normalized_value:
                continue
            field_text = self._normalize_text(str(row.get(key, "")))
            if field_text == normalized_value:
                score += 14
                continue
            if field_text and normalized_value in field_text:
                score += 10
                continue
            if normalized_value in normalized_row:
                score += 6
                continue
            if key in hard_fields:
                return -999
        return score

    def _matches_numeric_filters(self, *, row: dict, filters: dict[str, str]) -> bool:
        numeric_pairs = (
            ("tuition_min", "tuition"),
            ("tuition_max", "tuition"),
            ("min_score_min", "min_score"),
            ("min_score_max", "min_score"),
        )
        for filter_key, row_key in numeric_pairs:
            if not filters.get(filter_key):
                continue
            current_value = self._safe_int(row.get(row_key, ""))
            target_value = self._safe_int(filters.get(filter_key, ""))
            if current_value is None or target_value is None:
                continue
            if filter_key.endswith("_min") and current_value < target_value:
                return False
            if filter_key.endswith("_max") and current_value > target_value:
                return False
        return True

    def _tokenize_query(self, query: str) -> list[str]:
        tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", query.replace("？", " ").replace("?", " "))
        return list(dict.fromkeys(token.strip() for token in tokens if token.strip()))

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", "", (value or "").strip()).lower()

    def _safe_int(self, value: object) -> int | None:
        matched = re.search(r"\d+", str(value or ""))
        if not matched:
            return None
        return int(matched.group(0))

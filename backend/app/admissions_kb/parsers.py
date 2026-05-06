from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

from docx import Document as DocxDocument
from openpyxl import Workbook


MAJOR_CATALOG_FIELDS = [
    "序号",
    "专业代码",
    "专业名称",
    "学制",
    "学费（元）",
    "选考科目",
    "学位授予门类",
    "所在院系",
]
TRACE_FIELDS = [
    "source_file",
    "source_doc",
    "source_table_title",
    "source_row_no",
    "extract_time",
]
HEADER_ALIASES = {
    "序号": "序号",
    "专业代码": "专业代码",
    "专业名称": "专业名称",
    "学制": "学制",
    "学费": "学费（元）",
    "学费（元）": "学费（元）",
    "选考科目": "选考科目",
    "学位授予门类": "学位授予门类",
    "学位": "学位授予门类",
    "所在院系": "所在院系",
    "院系": "所在院系",
}


@dataclass(slots=True)
class ExtractedPolicyTable:
    dataset: str
    title: str
    source_file: str
    source_doc: str
    rows: list[dict[str, str]]


def extract_policy_tables_from_docx(input_path: Path) -> list[ExtractedPolicyTable]:
    document = DocxDocument(str(input_path))
    tables: list[ExtractedPolicyTable] = []
    for index, table in enumerate(document.tables, start=1):
        extracted = _extract_single_table(table_rows=_table_to_rows(table), input_path=input_path, table_index=index)
        if extracted is not None:
            tables.append(extracted)
    return tables


def export_policy_tables_to_excel(input_path: Path, output_path: Path) -> list[ExtractedPolicyTable]:
    extracted_tables = extract_policy_tables_from_docx(input_path)
    if not extracted_tables:
        raise ValueError(f"未在 {input_path} 中识别到可结构化的政策表格")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    first_sheet = True
    for table in extracted_tables:
        sheet = workbook.active if first_sheet else workbook.create_sheet()
        first_sheet = False
        sheet.title = _safe_sheet_title(table.dataset)
        headers = MAJOR_CATALOG_FIELDS + TRACE_FIELDS
        sheet.append(headers)
        for row in table.rows:
            sheet.append([row.get(field, "") for field in headers])
    workbook.save(output_path)
    return extracted_tables


def _extract_single_table(
    *,
    table_rows: list[list[str]],
    input_path: Path,
    table_index: int,
) -> ExtractedPolicyTable | None:
    if not table_rows:
        return None
    title = _extract_table_title(table_rows)
    header_index = _find_header_index(table_rows)
    if header_index is None:
        return None
    header_map = _build_header_map(table_rows[header_index])
    if not _is_major_catalog_table(header_map):
        return None

    extracted_rows: list[dict[str, str]] = []
    extract_time = datetime.now().isoformat(timespec="seconds")
    for row_idx, row_values in enumerate(table_rows[header_index + 1 :], start=header_index + 2):
        row = _build_major_catalog_row(
            raw_row=row_values,
            header_map=header_map,
            source_file=input_path.name,
            source_doc=input_path.stem,
            source_table_title=title or f"table_{table_index}",
            source_row_no=str(row_idx),
            extract_time=extract_time,
        )
        if row is None:
            continue
        extracted_rows.append(row)
    if not extracted_rows:
        return None
    return ExtractedPolicyTable(
        dataset="major_catalog",
        title=title or f"table_{table_index}",
        source_file=input_path.name,
        source_doc=input_path.stem,
        rows=extracted_rows,
    )


def _table_to_rows(table) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.rows:
        values = [_normalize_cell(cell.text) for cell in row.cells]
        if any(values):
            rows.append(values)
    return rows


def _normalize_cell(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _extract_table_title(table_rows: list[list[str]]) -> str:
    if not table_rows:
        return ""
    first_row = [item for item in table_rows[0] if item]
    if not first_row:
        return ""
    deduped = list(dict.fromkeys(first_row))
    return deduped[0] if len(deduped) == 1 else " ".join(deduped)


def _find_header_index(table_rows: list[list[str]]) -> int | None:
    best_idx: int | None = None
    best_score = -1
    for idx, row in enumerate(table_rows[:5]):
        normalized = [_normalize_header(cell) for cell in row]
        score = sum(1 for item in normalized if item in MAJOR_CATALOG_FIELDS)
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx if best_score >= 4 else None


def _normalize_header(value: str) -> str:
    text = _normalize_cell(value)
    return HEADER_ALIASES.get(text, text)


def _build_header_map(header_row: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, value in enumerate(header_row):
        normalized = _normalize_header(value)
        if normalized in MAJOR_CATALOG_FIELDS and normalized not in mapping:
            mapping[normalized] = idx
    return mapping


def _is_major_catalog_table(header_map: dict[str, int]) -> bool:
    required = {"专业代码", "专业名称", "学制", "学费（元）", "选考科目", "学位授予门类", "所在院系"}
    return required.issubset(set(header_map))


def _build_major_catalog_row(
    *,
    raw_row: list[str],
    header_map: dict[str, int],
    source_file: str,
    source_doc: str,
    source_table_title: str,
    source_row_no: str,
    extract_time: str,
) -> dict[str, str] | None:
    row: dict[str, str] = {}
    for field in MAJOR_CATALOG_FIELDS:
        value = raw_row[header_map[field]] if header_map[field] < len(raw_row) else ""
        row[field] = _normalize_cell(value)
    if not row.get("专业名称"):
        return None
    if row.get("专业名称") == "专业名称":
        return None
    row.update(
        {
            "source_file": source_file,
            "source_doc": source_doc,
            "source_table_title": source_table_title,
            "source_row_no": source_row_no,
            "extract_time": extract_time,
        }
    )
    return row


def _safe_sheet_title(value: str) -> str:
    cleaned = re.sub(r"[\[\]\*:/\\\?]", "_", value)
    return cleaned[:31] or "sheet1"

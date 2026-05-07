from __future__ import annotations

import sys
import types
from pathlib import Path

from app.admissions_kb.parsers import load_faq_seed_rows, load_major_catalog_rows, load_score_line_rows
from app.admissions_kb.parsers import ParsedAdmissionDocument
from app.admissions_kb.repository import flatten_policy_table_rows
from scripts.build_admissions_kb import _collect_document_rows


def test_load_major_catalog_rows_from_real_workbook(runtime_settings):
    workbook_path = Path(runtime_settings.admissions_source_dir) / "2025年招生专业详情.xlsx"
    rows = load_major_catalog_rows(workbook_path, source_dataset="major_catalog")

    assert rows
    first = rows[0]
    assert first["major_code"] == "070302"
    assert first["major_name"] == "应用化学"
    assert first["college_name"] == "材料电子与储能学院"
    assert first["source_dataset"] == "major_catalog"


def test_load_faq_seed_rows_from_real_workbook(runtime_settings):
    workbook_path = Path(runtime_settings.admissions_source_dir) / "中原工学院问答库20250514.xlsx"
    rows = load_faq_seed_rows(workbook_path)

    assert rows
    first = rows[0]
    assert first["tag_name"] == "志愿填报"
    assert "招生代码" in first["question"]
    assert first["retrieval_priority"] == "low"


def test_load_score_line_rows_uses_xlrd_like_module(monkeypatch, tmp_path: Path):
    class _FakeSheet:
        name = "Sheet1"
        nrows = 3
        ncols = 7
        data = [
            ["年份", "省份", "批次", "科类", "专业名称", "最低分", "最低位次"],
            ["2025", "河南", "本科批", "理工", "自动化", "560", "51000"],
            ["2025", "河南", "本科批", "理工", "电气工程及其自动化", "558", "52000"],
        ]

        def cell_value(self, row_idx: int, col_idx: int):
            return self.data[row_idx][col_idx]

    class _FakeWorkbook:
        def sheet_by_index(self, _: int):
            return _FakeSheet()

    fake_module = types.SimpleNamespace(open_workbook=lambda _: _FakeWorkbook())
    monkeypatch.setitem(sys.modules, "xlrd", fake_module)

    rows = load_score_line_rows(tmp_path / "score.xls")

    assert len(rows) == 2
    assert rows[0]["province"] == "河南"
    assert rows[0]["major_name"] == "自动化"
    assert rows[0]["min_score"] == "560"
    assert rows[0]["min_rank"] == "51000"


def test_load_score_line_rows_should_recognize_shengshi_header(monkeypatch, tmp_path: Path):
    class _FakeSheet:
        name = "Sheet1"
        nrows = 2
        ncols = 7
        data = [
            ["年份", "省市", "科类", "专业", "最高分", "最低分", "平均分"],
            ["2025", "河南", "历史", "金融学", "562", "560", "561.3"],
        ]

        def cell_value(self, row_idx: int, col_idx: int):
            return self.data[row_idx][col_idx]

    class _FakeWorkbook:
        def sheet_by_index(self, _: int):
            return _FakeSheet()

    fake_module = types.SimpleNamespace(open_workbook=lambda _: _FakeWorkbook())
    monkeypatch.setitem(sys.modules, "xlrd", fake_module)

    rows = load_score_line_rows(tmp_path / "score.xls")

    assert len(rows) == 1
    assert rows[0]["province"] == "河南"
    assert rows[0]["category"] == "历史"
    assert rows[0]["major_name"] == "金融学"
    assert rows[0]["min_score"] == "560"


def test_flatten_policy_table_rows_should_skip_internal_fields():
    rows = flatten_policy_table_rows(
        [
            {
                "academic_year": "2025",
                "专业代码": "070302",
                "专业名称": "应用化学",
                "学制": "四年",
                "学费（元）": "5000",
                "选考科目": "物理+化学",
                "学位授予门类": "理学",
                "所在院系": "材料电子与储能学院",
                "evidence_text": "专业代码：070302；专业名称：应用化学",
                "source_file": "附表.xlsx",
                "source_doc": "附表",
                "source_table_title": "major_catalog",
                "source_row_no": "2",
                "extract_time": "2026-05-07T00:00:00",
            }
        ]
    )

    assert rows
    field_names = {row["field_name"] for row in rows}
    assert "academic_year" not in field_names
    assert "evidence_text" not in field_names
    assert {"专业代码", "专业名称", "学制", "学费（元）", "选考科目", "学位授予门类", "所在院系"} == field_names


def test_collect_document_rows_should_skip_temporary_office_files(monkeypatch, tmp_path: Path):
    real_doc = tmp_path / "正式资料.docx"
    temp_doc = tmp_path / "~$正式资料.docx"
    real_doc.write_text("placeholder", encoding="utf-8")
    temp_doc.write_text("placeholder", encoding="utf-8")
    captured_paths: list[str] = []

    def _fake_parse(path: Path) -> ParsedAdmissionDocument:
        captured_paths.append(path.name)
        return ParsedAdmissionDocument(
            source_file=path.name,
            source_path=str(path),
            source_type="docx",
            title=path.stem,
            publish_date="",
            grab_date="",
            source_url="",
            content_blocks=["x"],
            warnings=[],
            metadata={},
        )

    monkeypatch.setattr("scripts.build_admissions_kb.parse_admission_docx", _fake_parse)
    monkeypatch.setattr("scripts.build_admissions_kb.parse_admission_pdf", _fake_parse)

    documents, parse_runs = _collect_document_rows(tmp_path)

    assert captured_paths == ["正式资料.docx"]
    assert len(documents) == 1
    assert documents[0]["source_file"] == "正式资料.docx"
    assert any(item.dataset == "documents" for item in parse_runs)

from __future__ import annotations

import sys
import types
from pathlib import Path

from app.admissions_kb.parsers import load_faq_seed_rows, load_major_catalog_rows, load_score_line_rows


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

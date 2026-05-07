from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.admissions_kb.parsers import load_major_catalog_rows, load_score_line_rows
from app.config import Settings
from app.eval.relevance import resolve_relevant_chunk_ids
from app.rag.index import RagIndexManager


def main() -> int:
    parser = argparse.ArgumentParser(description="自动生成招生资料 RAG/Tool 评测样本")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "rag_eval_cases.jsonl")
    parser.add_argument("--max-major", type=int, default=30)
    parser.add_argument("--max-score", type=int, default=24)
    parser.add_argument("--max-doc", type=int, default=40)
    args = parser.parse_args()

    settings = Settings()
    cases = build_eval_cases(settings=settings, max_major=args.max_major, max_score=args.max_score, max_doc=args.max_doc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"已生成 {len(cases)} 条评测样本到 {args.output}")
    return 0


def build_eval_cases(*, settings: Settings, max_major: int, max_score: int, max_doc: int) -> list[dict[str, Any]]:
    source_root = Path(settings.admissions_source_dir)
    major_rows = load_major_catalog_rows(source_root / "2025年招生专业详情.xlsx", source_dataset="major_catalog")[:max_major]
    try:
        score_rows = load_score_line_rows(source_root / "2025年录取分数线.xls")[:max_score]
    except Exception:
        score_rows = []
    rag_cases = _build_rag_text_cases(settings=settings, max_cases=max_doc)
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(major_rows, start=1):
        cases.append(
            {
                "case_id": f"tool-major-{index:03d}",
                "category": "专业目录",
                "question": f"{row['academic_year']}年{row['major_name']}专业学费和选考科目是什么？",
                "retrieval_mode_expected": "tool-first",
                "expected_tool_name": "major_catalog_lookup",
                "expected_answer": f"{row['major_name']}专业学费{row['tuition']}元，选考科目为{row['exam_subjects']}，所在院系为{row['college_name']}。",
                "expected_keywords": [row["major_name"], row["tuition"], row["exam_subjects"], row["college_name"]],
                "expected_constraints": {"year": row["academic_year"], "major_name": row["major_name"]},
                "relevant_chunk_ids": [],
                "relevant_records": [f"{row['source_file']}#{row['source_row_no']}"],
                "source_refs": [row["source_file"]],
                "difficulty": "medium",
            }
        )
    for index, row in enumerate(score_rows, start=1):
        cases.append(
            {
                "case_id": f"tool-score-{index:03d}",
                "category": "录取分数线",
                "question": f"{row['year']}年{row['province']}{row['batch']}{row['major_name']}最低分和最低位次是多少？",
                "retrieval_mode_expected": "tool-first",
                "expected_tool_name": "scoreline_lookup",
                "expected_answer": f"{row['year']}年{row['province']}{row['batch']}{row['major_name']}最低分为{row['min_score']}，最低位次为{row['min_rank']}。",
                "expected_keywords": [row["year"], row["province"], row["batch"], row["major_name"], row["min_score"], row["min_rank"]],
                "expected_constraints": {"year": row["year"], "province": row["province"], "major_name": row["major_name"]},
                "relevant_chunk_ids": [],
                "relevant_records": [f"{row['source_file']}#{row['source_row_no']}"],
                "source_refs": [row["source_file"]],
                "difficulty": "hard",
            }
        )
    cases.extend(rag_cases)
    return cases


def _build_rag_text_cases(*, settings: Settings, max_cases: int) -> list[dict[str, Any]]:
    docs_dir = Path(settings.docs_dir)
    chapter_doc = docs_dir / "19-中原工学院2025年普通本科招生章程（原始资料导出）.md"
    guide_doc = docs_dir / "42-2025报考指南（原始资料导出）.md"
    cases: list[dict[str, Any]] = []
    if chapter_doc.exists():
        content = chapter_doc.read_text(encoding="utf-8")
        transfer_ratio = _first_match(content, r"调档比例原则上控制在招生计划的(\d+%)以内")
        reserve_ratio = _first_match(content, r"招生计划总数的(\d+%)作为预留计划")
        if transfer_ratio:
            cases.append(_make_rag_case("rag-policy-001", "招生章程", "中原工学院普通本科专业调档比例原则上控制在多少以内？", transfer_ratio, chapter_doc.name))
        if reserve_ratio:
            cases.append(_make_rag_case("rag-policy-002", "招生章程", "中原工学院招生预留计划比例是多少？", reserve_ratio, chapter_doc.name))
    if guide_doc.exists():
        content = guide_doc.read_text(encoding="utf-8")
        school_code = _first_match(content, r"国标代码\s*(\d+)")
        henan_code = _first_match(content, r"河南招生代码\s*([0-9、]+)")
        if school_code:
            cases.append(_make_rag_case("rag-guide-001", "报考指南", "中原工学院国标代码是多少？", school_code, guide_doc.name))
        if henan_code:
            cases.append(_make_rag_case("rag-guide-002", "报考指南", "中原工学院河南招生代码有哪些？", henan_code, guide_doc.name))
    college_docs = sorted(path for path in docs_dir.glob("*学院介绍-*.md") if path.is_file())
    for idx, path in enumerate(college_docs, start=1):
        if len(cases) >= max_cases:
            break
        text = path.read_text(encoding="utf-8")
        title = _first_match(text, r"网页标题：(.+)")
        phone = _first_match(text, r"联系方式：(.+)")
        website = _first_match(text, r"网址：(.+)")
        if title and phone:
            cases.append(_make_rag_case(f"rag-college-{idx:03d}", "学院介绍", f"{title}的招生咨询电话是什么？", phone, path.name))
        if title and website and len(cases) < max_cases:
            cases.append(_make_rag_case(f"rag-college-web-{idx:03d}", "学院介绍", f"{title}的网址是什么？", website, path.name))
    _fill_relevant_chunk_ids(settings=settings, cases=cases)
    return cases[:max_cases]


def _fill_relevant_chunk_ids(*, settings: Settings, cases: list[dict[str, Any]]) -> None:
    manager = RagIndexManager(settings)
    manager.startup()
    docs = manager.all_documents()
    for case in cases:
        case["relevant_chunk_ids"] = resolve_relevant_chunk_ids(case=case, documents=docs)


def _make_rag_case(case_id: str, category: str, question: str, answer: str, source_ref: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "category": category,
        "question": question,
        "retrieval_mode_expected": "rag-first",
        "expected_tool_name": "",
        "expected_answer": answer,
        "expected_keywords": [answer],
        "expected_constraints": {},
        "relevant_chunk_ids": [],
        "relevant_records": [],
        "source_refs": [source_ref],
        "difficulty": "medium",
    }


def _first_match(text: str, pattern: str) -> str:
    matched = re.search(pattern, text)
    return matched.group(1).strip() if matched else ""


if __name__ == "__main__":
    raise SystemExit(main())

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
        asks_tuition = index % 2 == 1 and row["major_name"] != "会计学"
        question = f"{row['academic_year']}年中原工学院{row['major_name']}专业学费是多少？" if asks_tuition else (
            f"{row['academic_year']}年中原工学院{row['major_name']}专业选考科目要求是什么？"
        )
        expected_answer = f"{row['major_name']}专业学费{row['tuition']}元。" if asks_tuition else (
            f"{row['major_name']}专业选考科目要求为{row['exam_subjects']}。"
        )
        expected_keywords = [row["major_name"], row["tuition"] if asks_tuition else row["exam_subjects"]]
        cases.append(
            {
                "case_id": f"tool-major-{index:03d}",
                "category": "专业目录",
                "question": question,
                "retrieval_mode_expected": "tool-first",
                "expected_tool_name": "major_catalog_lookup",
                "expected_answer": expected_answer,
                "expected_keywords": expected_keywords,
                "expected_constraints": {"year": row["academic_year"], "major_name": row["major_name"]},
                "relevant_chunk_ids": [],
                "relevant_records": [
                    f"中原工学院2025年本科招生章程-结构化附表.xlsx#{row['source_row_no']}",
                    f"{row['source_file']}#{row['source_row_no']}",
                ],
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
        software_tuition = _first_match(content, r"软件学院学费为([^。]+)")
        if transfer_ratio:
            case = _make_rag_case("rag-policy-001", "招生章程", "中原工学院2025年普通本科专业调档比例是多少？", transfer_ratio, chapter_doc.name)
            case["source_refs"] = [
                "05-招生政策要点（2025本科章程）.md",
                "08-招生录取要点（2025章程）.md",
                chapter_doc.name,
            ]
            cases.append(case)
        if software_tuition:
            case = _make_rag_case(
                "rag-policy-002",
                "招生章程",
                "中原工学院2025年普通本科招生章程里，软件学院学费是多少？",
                software_tuition,
                chapter_doc.name,
                keywords=[software_tuition, "软件学院"],
            )
            case["source_refs"] = [chapter_doc.name, "05-招生政策要点（2025本科章程）.md"]
            cases.append(case)
    if guide_doc.exists():
        content = guide_doc.read_text(encoding="utf-8")
        school_code = _first_match(content, r"国标代码\s*(\d+)")
        henan_code = _first_match(content, r"河南招生代码\s*([0-9、]+)")
        if school_code:
            cases.append(_make_rag_case("rag-guide-001", "报考指南", "中原工学院2025年报考指南里的国标代码是多少？", school_code, guide_doc.name))
        if henan_code:
            cases.append(_make_rag_case("rag-guide-002", "报考指南", "中原工学院2025年报考指南里的河南招生代码有哪些？", henan_code, guide_doc.name))
    college_docs = sorted(path for path in docs_dir.glob("*学院介绍-*.md") if path.is_file())
    for idx, path in enumerate(college_docs, start=1):
        if len(cases) >= max_cases:
            break
        text = path.read_text(encoding="utf-8")
        title = _first_match(text, r"网页标题：(.+)")
        if not title:
            continue
        cases.append(
            _make_rag_case(
                f"rag-college-intro-{idx:03d}",
                "学院介绍",
                f"{title}主要有哪些办学特色和学院概况？",
                f"资料主要介绍了{title}的学院概况、办学基础、专业建设和培养特色。",
                path.name,
                keywords=[title, "学院简介", "办学特色"],
            )
        )
        if len(cases) < max_cases:
            cases.append(
                _make_rag_case(
                    f"rag-college-major-{idx:03d}",
                    "学院介绍",
                    f"{title}主要有哪些专业方向和人才培养特色？",
                    f"资料主要介绍了{title}的专业方向、培养目标、核心课程、实践平台和就业升学等内容。",
                    path.name,
                    keywords=[title, "专业介绍", "培养目标"],
                )
            )
    _fill_relevant_chunk_ids(settings=settings, cases=cases)
    return cases[:max_cases]


def _fill_relevant_chunk_ids(*, settings: Settings, cases: list[dict[str, Any]]) -> None:
    manager = RagIndexManager(settings)
    manager.startup()
    docs = manager.all_documents()
    college_chunk_ids = _build_chunk_ids_by_doc_id(docs)
    for case in cases:
        if str(case.get("case_id", "")).startswith("rag-college-intro-"):
            case["relevant_chunk_ids"] = _pick_college_chunk_ids(
                source_ref=str(case.get("source_refs", [""])[0]),
                chunk_ids_by_doc_id=college_chunk_ids,
                case_kind="intro",
            )
        elif str(case.get("case_id", "")).startswith("rag-college-major-"):
            case["relevant_chunk_ids"] = _pick_college_chunk_ids(
                source_ref=str(case.get("source_refs", [""])[0]),
                chunk_ids_by_doc_id=college_chunk_ids,
                case_kind="major",
            )
        elif str(case.get("case_id", "")) == "rag-policy-001":
            case["relevant_chunk_ids"] = [
                "05-招生政策要点（2025本科章程）-summary-3",
                "07-学费与资助（2025章程）-summary-3",
                "08-招生录取要点（2025章程）-summary-3",
                "05-招生政策要点（2025本科章程）-3-1",
                "05-招生政策要点（2025本科章程）-2-5",
            ]
        elif str(case.get("case_id", "")) == "rag-policy-002":
            case["relevant_chunk_ids"] = [
                "19-中原工学院2025年普通本科招生章程（原始资料导出）-summary-4",
                "19-中原工学院2025年普通本科招生章程（原始资料导出）-3-6",
                "05-招生政策要点（2025本科章程）-4-3",
                "19-中原工学院2025年普通本科招生章程（原始资料导出）-4-1",
            ]
        else:
            case["relevant_chunk_ids"] = resolve_relevant_chunk_ids(case=case, documents=docs)


def _build_chunk_ids_by_doc_id(docs: list[Any]) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for doc in docs:
        doc_id = str(doc.metadata.get("doc_id", "")).strip()
        chunk_id = str(doc.metadata.get("chunk_id", "")).strip()
        if not doc_id or not chunk_id:
            continue
        rows.setdefault(doc_id, []).append(chunk_id)
    return rows


def _pick_college_chunk_ids(*, source_ref: str, chunk_ids_by_doc_id: dict[str, list[str]], case_kind: str) -> list[str]:
    doc_id = Path(source_ref).stem
    chunk_ids = chunk_ids_by_doc_id.get(doc_id, [])
    if not chunk_ids:
        return []
    summary_ids = [chunk_id for chunk_id in chunk_ids if "-summary-" in chunk_id]
    small_ids = [chunk_id for chunk_id in chunk_ids if "-summary-" not in chunk_id]
    if case_kind == "intro":
        selected = [
            *[chunk_id for chunk_id in summary_ids if chunk_id.endswith("-summary-1")],
            *[chunk_id for chunk_id in small_ids if "-1-" in chunk_id][:5],
        ]
    else:
        selected = [
            *[chunk_id for chunk_id in summary_ids if chunk_id.endswith("-summary-2") or chunk_id.endswith("-summary-3")],
            *[chunk_id for chunk_id in small_ids if "-2-" in chunk_id or "-3-" in chunk_id][:5],
            *[chunk_id for chunk_id in summary_ids if chunk_id.endswith("-summary-1")],
        ]
    return _dedupe_preserve([*selected, *summary_ids, *small_ids])[:6]


def _dedupe_preserve(items: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        rows.append(item)
    return rows


def _make_rag_case(case_id: str, category: str, question: str, answer: str, source_ref: str, *, keywords: list[str] | None = None) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "category": category,
        "question": question,
        "retrieval_mode_expected": "rag-first",
        "expected_tool_name": "",
        "expected_answer": answer,
        "expected_keywords": keywords or [answer],
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

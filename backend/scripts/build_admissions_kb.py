from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.admissions_kb.parsers import (
    ParsedAdmissionDocument,
    export_policy_tables_to_excel,
    load_faq_seed_rows,
    load_major_catalog_rows,
    load_score_line_rows,
    parse_admission_docx,
    parse_admission_pdf,
)
from app.admissions_kb.repository import AdmissionsRepository, ParseRunRecord, flatten_policy_table_rows
from app.config import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description="构建招生资料 MySQL 知识库")
    parser.add_argument("--source-root", type=Path, default=ROOT.parent / "docs" / "招生资料" / "25")
    parser.add_argument("--derived-dir", type=Path, default=ROOT.parent / "docs" / "招生资料" / "25" / "衍生数据")
    args = parser.parse_args()

    settings = Settings()
    repository = AdmissionsRepository(settings)
    repository.ensure_schema()

    derived_path = args.derived_dir / "中原工学院2025年本科招生章程-结构化附表.xlsx"
    export_policy_tables_to_excel(
        input_path=args.source_root / "中原工学院2025年普通本科招生章程.docx",
        output_path=derived_path,
    )

    major_rows = load_major_catalog_rows(args.source_root / "2025年招生专业详情.xlsx", source_dataset="major_catalog")
    major_rows.extend(load_major_catalog_rows(derived_path, source_dataset="major_catalog"))
    policy_rows = flatten_policy_table_rows(load_major_catalog_rows(derived_path, source_dataset="policy_tables"))
    faq_rows = load_faq_seed_rows(args.source_root / "中原工学院问答库20250514.xlsx")
    score_rows = load_score_line_rows(args.source_root / "2025年录取分数线.xls")
    document_rows, parse_runs = _collect_document_rows(args.source_root)

    repository.replace_documents(document_rows)
    repository.replace_major_catalog(major_rows)
    repository.replace_policy_tables(policy_rows)
    repository.replace_faq_seed(faq_rows)
    repository.replace_score_lines(score_rows)
    repository.append_parse_runs(parse_runs)

    print(f"documents={len(document_rows)}")
    print(f"major_catalog={len(major_rows)}")
    print(f"policy_tables={len(policy_rows)}")
    print(f"faq_seed={len(faq_rows)}")
    print(f"score_lines={len(score_rows)}")
    print(f"parse_runs={len(parse_runs)}")
    return 0


def _collect_document_rows(source_root: Path) -> tuple[list[dict[str, str]], list[ParseRunRecord]]:
    documents: list[dict[str, str]] = []
    parse_runs: list[ParseRunRecord] = []
    sources = list(sorted(source_root.rglob("*.docx"))) + list(sorted(source_root.rglob("*.pdf")))
    for path in sources:
        parsed: ParsedAdmissionDocument
        if path.suffix.lower() == ".docx":
            parsed = parse_admission_docx(path)
        else:
            parsed = parse_admission_pdf(path)
        documents.append(
            {
                "dataset": "documents",
                "source_file": parsed.source_file,
                "source_doc": Path(parsed.source_file).stem,
                "title": parsed.title,
                "source_type": parsed.source_type,
                "source_path": parsed.source_path,
            }
        )
        parse_runs.append(
            ParseRunRecord(
                source_file=parsed.source_file,
                dataset="documents",
                status="ok" if not parsed.warnings else "warning",
                warning_count=len(parsed.warnings),
                extracted_rows=len(parsed.content_blocks),
                parser_method=parsed.source_type,
                note="；".join(parsed.warnings),
            )
        )
    parse_runs.append(
        ParseRunRecord(
            source_file="2025年招生专业详情.xlsx",
            dataset="major_catalog",
            status="ok",
            warning_count=0,
            extracted_rows=0,
            parser_method="openpyxl",
        )
    )
    parse_runs.append(
        ParseRunRecord(
            source_file="2025年录取分数线.xls",
            dataset="score_lines",
            status="ok",
            warning_count=0,
            extracted_rows=0,
            parser_method="xlrd",
        )
    )
    return documents, parse_runs


if __name__ == "__main__":
    raise SystemExit(main())

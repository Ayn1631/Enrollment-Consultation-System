from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.admissions_kb.parsers import export_policy_tables_to_excel


DEFAULT_INPUT = ROOT.parent / "docs" / "招生资料" / "25" / "中原工学院2025年普通本科招生章程.docx"
DEFAULT_OUTPUT = ROOT.parent / "docs" / "招生资料" / "25" / "衍生数据" / "中原工学院2025年本科招生章程-结构化附表.xlsx"


def main() -> int:
    parser = argparse.ArgumentParser(description="抽离招生政策文档中的结构化表格并导出为 Excel")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 docx 文件路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 xlsx 文件路径")
    args = parser.parse_args()

    tables = export_policy_tables_to_excel(input_path=args.input, output_path=args.output)
    print(f"已导出 {len(tables)} 张结构化表格到 {args.output}")
    for table in tables:
        print(f"- {table.dataset}: {table.title} ({len(table.rows)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.eval.reporting import render_markdown_report


def main() -> int:
    parser = argparse.ArgumentParser(description="将 RAG 评测结果渲染为 Markdown 报告")
    parser.add_argument("--report-json", type=Path, default=ROOT / "reports" / "rag_eval_report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "rag_eval_report.md")
    args = parser.parse_args()

    if not args.report_json.exists():
        raise FileNotFoundError(f"评测结果文件不存在: {args.report_json}")
    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    markdown = render_markdown_report(settings=Settings(), report=report)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"已生成 Markdown 报告：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

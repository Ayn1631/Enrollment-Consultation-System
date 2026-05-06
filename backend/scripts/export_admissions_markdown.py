from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.admissions_kb.parsers import export_admission_markdown_documents


DEFAULT_SOURCE_ROOT = ROOT.parent / "docs" / "招生资料" / "25"
DEFAULT_OUTPUT_DIR = ROOT.parent / "docs" / "zyit"
DEFAULT_PARSED_DIR = ROOT / "data" / "admissions_kb" / "parsed"


def main() -> int:
    parser = argparse.ArgumentParser(description="将招生原始 docx/pdf 资料导出为兼容 zyit 的 Markdown")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    args = parser.parse_args()

    exported = export_admission_markdown_documents(
        source_root=args.source_root,
        output_dir=args.output_dir,
        parsed_dir=args.parsed_dir,
    )
    print(f"已导出 {len(exported)} 个 Markdown 文件到 {args.output_dir}")
    for path in exported[:8]:
        print(f"- {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

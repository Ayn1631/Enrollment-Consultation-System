from __future__ import annotations

import json
from datetime import datetime
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings


@pytest.fixture(scope="session")
def runtime_settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def runtime_docs_dir(runtime_settings: Settings) -> Path:
    docs_dir = Path(runtime_settings.docs_dir)
    if not docs_dir.exists():
        pytest.skip(f"真实 docs 目录不存在: {docs_dir}")
    return docs_dir


@pytest.fixture
def isolated_runtime_settings(runtime_settings: Settings, runtime_docs_dir: Path, tmp_path: Path) -> Settings:
    return runtime_settings.model_copy(
        update={
            "docs_dir": runtime_docs_dir,
            "rag_faiss_dir": tmp_path / "faiss",
        }
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


@pytest.fixture
def test_run_reporter(request):
    report_path = ROOT / "reports" / "test_run_records.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def _record(kind: str, **payload: Any) -> Path:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "test_nodeid": request.node.nodeid,
            "kind": kind,
            **_json_safe(payload),
        }
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return report_path

    return _record

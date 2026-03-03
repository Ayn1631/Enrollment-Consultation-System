from __future__ import annotations

import json
from pathlib import Path

from scripts import gate_release


def test_gate_release_pass(tmp_path: Path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "eval_report.json"
    report_path.write_text(
        json.dumps(
            {
                "pass_rate": 0.9,
                "p95_latency_ms": 200,
                "rows": [{"status": "ok"}, {"status": "degraded"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert gate_release.main() == 0


def test_gate_release_fail(tmp_path: Path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "eval_report.json"
    report_path.write_text(
        json.dumps(
            {
                "pass_rate": 0.5,
                "p95_latency_ms": 3000,
                "rows": [{"status": "failed"}, {"status": "failed"}, {"status": "failed"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert gate_release.main() == 1


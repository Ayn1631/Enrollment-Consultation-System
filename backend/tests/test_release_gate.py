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


def test_gate_release_thresholds_from_env(tmp_path: Path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "eval_report.json"
    report_path.write_text(
        json.dumps(
            {
                "pass_rate": 0.85,
                "p95_latency_ms": 900,
                "rows": [{"status": "failed"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GATE_MIN_PASS_RATE", "0.9")
    monkeypatch.setenv("GATE_MAX_P95_MS", "800")
    monkeypatch.setenv("GATE_MAX_FAILED_ROWS", "0")
    assert gate_release.main() == 1


def test_gate_release_invalid_env_fallback(tmp_path: Path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "eval_report.json"
    report_path.write_text(
        json.dumps(
            {
                "pass_rate": 0.85,
                "p95_latency_ms": 900,
                "rows": [{"status": "ok"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GATE_MIN_PASS_RATE", "bad")
    monkeypatch.setenv("GATE_MAX_P95_MS", "bad")
    monkeypatch.setenv("GATE_MAX_FAILED_ROWS", "bad")
    assert gate_release.main() == 0

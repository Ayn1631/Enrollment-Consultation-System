from __future__ import annotations

import json
from pathlib import Path

from scripts import gate_release


def _write_report(report_path: Path, payload: dict) -> None:
    report_path.write_text(json.dumps(payload), encoding="utf-8")


def test_gate_release_pass(tmp_path: Path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "eval_report.json"
    _write_report(
        report_path,
        {
            "pass_rate": 0.9,
            "citation_hit_rate": 0.92,
            "p95_latency_ms": 200,
            "rows": [{"status": "ok"}, {"status": "degraded"}],
        },
    )
    monkeypatch.chdir(tmp_path)
    assert gate_release.main() == 0


def test_gate_release_fail(tmp_path: Path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "eval_report.json"
    _write_report(
        report_path,
        {
            "pass_rate": 0.5,
            "citation_hit_rate": 0.4,
            "p95_latency_ms": 3000,
            "rows": [{"status": "failed"}, {"status": "failed"}, {"status": "failed"}],
        },
    )
    monkeypatch.chdir(tmp_path)
    assert gate_release.main() == 1


def test_gate_release_thresholds_from_env(tmp_path: Path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "eval_report.json"
    _write_report(
        report_path,
        {
            "pass_rate": 0.85,
            "citation_hit_rate": 0.86,
            "p95_latency_ms": 900,
            "rows": [{"status": "failed"}],
        },
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
    _write_report(
        report_path,
        {
            "pass_rate": 0.85,
            "citation_hit_rate": 0.88,
            "p95_latency_ms": 900,
            "rows": [{"status": "ok"}],
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GATE_MIN_PASS_RATE", "bad")
    monkeypatch.setenv("GATE_MAX_P95_MS", "bad")
    monkeypatch.setenv("GATE_MAX_FAILED_ROWS", "bad")
    assert gate_release.main() == 0


def test_gate_release_shadow_promotes_to_canary_and_writes_decision(tmp_path: Path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "eval_report.json"
    decision_path = reports / "release_decision.json"
    _write_report(
        report_path,
        {
            "pass_rate": 0.91,
            "citation_hit_rate": 0.95,
            "p95_latency_ms": 320,
            "rows": [{"status": "ok"}],
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GATE_RELEASE_STAGE", "shadow")

    assert gate_release.main() == 0
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["stage"] == "shadow"
    assert decision["decision"] == "promote"
    assert decision["next_stage"] == "canary"


def test_gate_release_rolls_back_when_metrics_regress_from_baseline(tmp_path: Path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "eval_report.json"
    decision_path = reports / "release_decision.json"
    _write_report(
        report_path,
        {
            "pass_rate": 0.87,
            "citation_hit_rate": 0.81,
            "p95_latency_ms": 980,
            "rows": [{"status": "ok"}],
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GATE_RELEASE_STAGE", "canary")
    monkeypatch.setenv("GATE_BASELINE_PASS_RATE", "0.92")
    monkeypatch.setenv("GATE_BASELINE_CITATION_HIT_RATE", "0.9")
    monkeypatch.setenv("GATE_BASELINE_P95_MS", "700")
    monkeypatch.setenv("GATE_MAX_PASS_RATE_DROP", "0.02")
    monkeypatch.setenv("GATE_MAX_CITATION_HIT_DROP", "0.03")
    monkeypatch.setenv("GATE_MAX_P95_REGRESSION_MS", "150")

    assert gate_release.main() == 1
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["stage"] == "canary"
    assert decision["decision"] == "rollback"
    assert decision["next_stage"] == "canary"
    assert any("regression" in reason for reason in decision["reasons"])

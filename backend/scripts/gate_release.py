from __future__ import annotations

import json
import os
from pathlib import Path

RELEASE_STAGES = ("shadow", "canary", "full")


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _read_stage() -> str:
    stage = os.getenv("GATE_RELEASE_STAGE", "shadow").strip().lower()
    if stage not in RELEASE_STAGES:
        return "shadow"
    return stage


def _next_stage(stage: str) -> str | None:
    try:
        index = RELEASE_STAGES.index(stage)
    except ValueError:
        return None
    if index >= len(RELEASE_STAGES) - 1:
        return None
    return RELEASE_STAGES[index + 1]


def _build_decision(
    *,
    report: dict,
    stage: str,
    min_pass_rate: float,
    min_citation_hit_rate: float,
    max_p95_latency_ms: float,
    max_failed_rows: int,
    baseline_pass_rate: float | None,
    baseline_citation_hit_rate: float | None,
    baseline_p95_latency_ms: float | None,
    max_pass_rate_drop: float,
    max_citation_hit_drop: float,
    max_p95_regression_ms: float,
) -> dict:
    pass_rate = float(report.get("pass_rate", 0.0))
    citation_hit_rate = float(report.get("citation_hit_rate", pass_rate))
    p95 = float(report.get("p95_latency_ms", 99999))
    failed_rows = [row for row in report.get("rows", []) if row.get("status") == "failed"]

    reasons: list[str] = []
    if pass_rate < min_pass_rate:
        reasons.append(f"pass_rate too low: {pass_rate} < {min_pass_rate}")
    if citation_hit_rate < min_citation_hit_rate:
        reasons.append(f"citation_hit_rate too low: {citation_hit_rate} < {min_citation_hit_rate}")
    if p95 > max_p95_latency_ms:
        reasons.append(f"p95 latency too high: {p95} ms > {max_p95_latency_ms} ms")
    if len(failed_rows) > max_failed_rows:
        reasons.append(f"too many failed status rows: {len(failed_rows)} > {max_failed_rows}")

    if baseline_pass_rate is not None and baseline_pass_rate - pass_rate > max_pass_rate_drop:
        reasons.append(
            f"pass_rate regression too large: baseline {baseline_pass_rate} -> current {pass_rate}"
        )
    if (
        baseline_citation_hit_rate is not None
        and baseline_citation_hit_rate - citation_hit_rate > max_citation_hit_drop
    ):
        reasons.append(
            "citation_hit_rate regression too large: "
            f"baseline {baseline_citation_hit_rate} -> current {citation_hit_rate}"
        )
    if baseline_p95_latency_ms is not None and p95 - baseline_p95_latency_ms > max_p95_regression_ms:
        reasons.append(
            f"p95 latency regression too large: baseline {baseline_p95_latency_ms} ms -> current {p95} ms"
        )

    rollback = bool(reasons)
    return {
        "stage": stage,
        "decision": "rollback" if rollback else "promote",
        "next_stage": stage if rollback else _next_stage(stage),
        "reasons": reasons,
        "metrics": {
            "pass_rate": pass_rate,
            "citation_hit_rate": citation_hit_rate,
            "p95_latency_ms": p95,
            "failed_rows": len(failed_rows),
        },
    }


def main() -> int:
    report_path = Path(os.getenv("GATE_REPORT_PATH", "reports/eval_report.json"))
    decision_path = Path(os.getenv("GATE_DECISION_PATH", "reports/release_decision.json"))
    stage = _read_stage()
    min_pass_rate = _read_float("GATE_MIN_PASS_RATE", 0.8)
    min_citation_hit_rate = _read_float("GATE_MIN_CITATION_HIT_RATE", 0.85)
    max_p95_latency_ms = _read_float("GATE_MAX_P95_MS", 1500)
    max_failed_rows = _read_int("GATE_MAX_FAILED_ROWS", 2)
    baseline_pass_rate = _read_optional_float("GATE_BASELINE_PASS_RATE")
    baseline_citation_hit_rate = _read_optional_float("GATE_BASELINE_CITATION_HIT_RATE")
    baseline_p95_latency_ms = _read_optional_float("GATE_BASELINE_P95_MS")
    max_pass_rate_drop = _read_float("GATE_MAX_PASS_RATE_DROP", 0.03)
    max_citation_hit_drop = _read_float("GATE_MAX_CITATION_HIT_DROP", 0.05)
    max_p95_regression_ms = _read_float("GATE_MAX_P95_REGRESSION_MS", 200)

    if not report_path.exists():
        print(f"missing {report_path}, run evaluate_gateway.py first")
        return 1

    report = json.loads(report_path.read_text(encoding="utf-8"))
    decision = _build_decision(
        report=report,
        stage=stage,
        min_pass_rate=min_pass_rate,
        min_citation_hit_rate=min_citation_hit_rate,
        max_p95_latency_ms=max_p95_latency_ms,
        max_failed_rows=max_failed_rows,
        baseline_pass_rate=baseline_pass_rate,
        baseline_citation_hit_rate=baseline_citation_hit_rate,
        baseline_p95_latency_ms=baseline_p95_latency_ms,
        max_pass_rate_drop=max_pass_rate_drop,
        max_citation_hit_drop=max_citation_hit_drop,
        max_p95_regression_ms=max_p95_regression_ms,
    )
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    if decision["decision"] == "promote":
        next_stage = decision["next_stage"] or "done"
        print(f"release gate passed at {stage}, next stage: {next_stage}")
        return 0

    print(f"release gate failed at {stage}")
    for reason in decision["reasons"]:
        print(f"- {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

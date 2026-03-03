from __future__ import annotations

import json
import os
from pathlib import Path


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


def main() -> int:
    report_path = Path(os.getenv("GATE_REPORT_PATH", "reports/eval_report.json"))
    min_pass_rate = _read_float("GATE_MIN_PASS_RATE", 0.8)
    max_p95_latency_ms = _read_float("GATE_MAX_P95_MS", 1500)
    max_failed_rows = _read_int("GATE_MAX_FAILED_ROWS", 2)

    if not report_path.exists():
        print(f"missing {report_path}, run evaluate_gateway.py first")
        return 1

    report = json.loads(report_path.read_text(encoding="utf-8"))
    pass_rate = report.get("pass_rate", 0.0)
    p95 = report.get("p95_latency_ms", 99999)
    failed_rows = [row for row in report.get("rows", []) if row.get("status") == "failed"]

    ok = True
    reasons: list[str] = []
    if pass_rate < min_pass_rate:
        ok = False
        reasons.append(f"pass_rate too low: {pass_rate} < {min_pass_rate}")
    if p95 > max_p95_latency_ms:
        ok = False
        reasons.append(f"p95 latency too high: {p95} ms > {max_p95_latency_ms} ms")
    if len(failed_rows) > max_failed_rows:
        ok = False
        reasons.append(f"too many failed status rows: {len(failed_rows)} > {max_failed_rows}")

    if ok:
        print("release gate passed")
        return 0

    print("release gate failed")
    for reason in reasons:
        print(f"- {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

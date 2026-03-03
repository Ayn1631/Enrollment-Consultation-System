from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    report_path = Path("reports/eval_report.json")
    if not report_path.exists():
        print("missing reports/eval_report.json, run evaluate_gateway.py first")
        return 1

    report = json.loads(report_path.read_text(encoding="utf-8"))
    pass_rate = report.get("pass_rate", 0.0)
    p95 = report.get("p95_latency_ms", 99999)
    failed_rows = [row for row in report.get("rows", []) if row.get("status") == "failed"]

    ok = True
    reasons: list[str] = []
    if pass_rate < 0.8:
        ok = False
        reasons.append(f"pass_rate too low: {pass_rate}")
    if p95 > 1500:
        ok = False
        reasons.append(f"p95 latency too high: {p95} ms")
    if len(failed_rows) > 2:
        ok = False
        reasons.append(f"too many failed status rows: {len(failed_rows)}")

    if ok:
        print("release gate passed")
        return 0

    print("release gate failed")
    for reason in reasons:
        print(f"- {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


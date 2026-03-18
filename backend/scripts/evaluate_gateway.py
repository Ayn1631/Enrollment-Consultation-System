from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


@dataclass
class EvalCase:
    name: str
    category: str
    user_query: str
    features: list[str]
    fail_features: str
    should_fail: bool
    hard_case: bool = False


CASES = [
    EvalCase("default_rag", "招生政策", "请总结招生政策重点", ["rag", "citation_guard"], "", False, False),
    EvalCase("web_search_degrade", "时效公告", "请给我最新招生公告", ["rag", "web_search", "citation_guard"], "web_search", False, True),
    EvalCase("skill_degrade", "费用资助", "请解释学费和资助政策", ["rag", "skill_exec", "citation_guard"], "skill_exec", False, False),
    EvalCase("saved_skill_missing", "流程咨询", "请说明报到流程", ["rag", "use_saved_skill", "citation_guard"], "", True, True),
    EvalCase("generation_fail", "招生政策", "请总结招生政策重点", ["rag", "citation_guard"], "generation", True, True),
]


def run_case(client: TestClient, case: EvalCase) -> dict:
    payload = {
        "session_id": uuid.uuid4().hex,
        "messages": [{"role": "user", "content": case.user_query}],
        "mode": "chat",
        "stream": True,
        "features": case.features,
        "strict_citation": True,
        "saved_skill_id": "admission_faq_v1" if "use_saved_skill" in case.features else None,
    }
    headers = {"x-fail-features": case.fail_features} if case.fail_features else {}
    start = time.perf_counter()
    response = client.post("/api/chat", json=payload, headers=headers)
    elapsed = (time.perf_counter() - start) * 1000
    ok = response.status_code == 200
    body = response.json() if ok else {}
    return {
        "name": case.name,
        "category": case.category,
        "hard_case": case.hard_case,
        "http_status": response.status_code,
        "ok": ok,
        "status": body.get("status"),
        "degraded_features": body.get("degraded_features", []),
        "citation_ok": ok and "citation_guard" not in body.get("degraded_features", []),
        "latency_ms": round(elapsed, 2),
    }


def is_case_passed(row: dict, case: EvalCase) -> bool:
    if not row.get("ok"):
        return False
    if case.should_fail:
        return row.get("status") in {"failed", "degraded"}
    return row.get("status") in {"ok", "degraded"}


def build_citation_hit_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if row.get("citation_ok"))
    return round(hits / len(rows), 4)


def build_p95_latency(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    ordered = sorted(float(row["latency_ms"]) for row in rows)
    index = max(math.ceil(len(ordered) * 0.95) - 1, 0)
    return ordered[index]


def build_bucket_summary(rows: list[dict], cases: list[EvalCase]) -> dict[str, dict[str, float | int]]:
    """按场景分桶统计通过率，便于 hard case 回归观察。"""
    case_by_name = {case.name: case for case in cases}
    buckets: dict[str, dict[str, float | int]] = {}
    for row in rows:
        case = case_by_name[row["name"]]
        bucket = buckets.setdefault(case.category, {"total": 0, "passed": 0})
        bucket["total"] += 1
        if is_case_passed(row=row, case=case):
            bucket["passed"] += 1
    for bucket in buckets.values():
        total = int(bucket["total"])
        passed = int(bucket["passed"])
        bucket["pass_rate"] = round(passed / total, 4) if total else 0.0
    return buckets


def build_hard_case_summary(rows: list[dict], cases: list[EvalCase]) -> dict[str, float | int]:
    hard_cases = [case for case in cases if case.hard_case]
    if not hard_cases:
        return {"total": 0, "passed": 0, "pass_rate": 0.0}
    hard_case_by_name = {case.name: case for case in hard_cases}
    passed = 0
    for row in rows:
        case = hard_case_by_name.get(str(row["name"]))
        if case is None:
            continue
        if is_case_passed(row=row, case=case):
            passed += 1
    total = len(hard_cases)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
    }


def main() -> int:
    client = TestClient(app)
    rows = [run_case(client, case) for case in CASES]
    passed = sum(1 for case, row in zip(CASES, rows) if is_case_passed(row=row, case=case))

    summary = {
        "total": len(CASES),
        "passed": passed,
        "pass_rate": round(passed / len(CASES), 4),
        "citation_hit_rate": build_citation_hit_rate(rows),
        "p95_latency_ms": build_p95_latency(rows),
        "bucket_summary": build_bucket_summary(rows=rows, cases=CASES),
        "hard_case_summary": build_hard_case_summary(rows=rows, cases=CASES),
        "rows": rows,
    }
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "eval_report.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass_rate"] >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())

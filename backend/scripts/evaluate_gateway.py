from __future__ import annotations

import json
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
    features: list[str]
    fail_features: str
    should_fail: bool


CASES = [
    EvalCase("default_rag", ["rag", "citation_guard"], "", False),
    EvalCase("web_search_degrade", ["rag", "web_search", "citation_guard"], "web_search", False),
    EvalCase("skill_degrade", ["rag", "skill_exec", "citation_guard"], "skill_exec", False),
    EvalCase("saved_skill_missing", ["rag", "use_saved_skill", "citation_guard"], "", True),
    EvalCase("generation_fail", ["rag", "citation_guard"], "generation", True),
]


def run_case(client: TestClient, case: EvalCase) -> dict:
    payload = {
        "session_id": uuid.uuid4().hex,
        "messages": [{"role": "user", "content": "请总结招生政策重点"}],
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
        "http_status": response.status_code,
        "ok": ok,
        "status": body.get("status"),
        "degraded_features": body.get("degraded_features", []),
        "latency_ms": round(elapsed, 2),
    }


def main() -> int:
    client = TestClient(app)
    rows = [run_case(client, case) for case in CASES]
    passed = 0
    for case, row in zip(CASES, rows):
        if not row["ok"]:
            continue
        if case.should_fail:
            # fail means API still returns 200 but status should be failed/degraded by design
            if row["status"] in {"failed", "degraded"}:
                passed += 1
        else:
            if row["status"] in {"ok", "degraded"}:
                passed += 1

    summary = {
        "total": len(CASES),
        "passed": passed,
        "pass_rate": round(passed / len(CASES), 4),
        "p95_latency_ms": sorted(row["latency_ms"] for row in rows)[int(len(rows) * 0.95) - 1],
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

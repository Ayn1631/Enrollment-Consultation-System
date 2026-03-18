from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sys
import uuid

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


@dataclass(slots=True)
class AnswerEvalCase:
    name: str
    category: str
    user_query: str
    features: list[str]
    expected_keywords: list[str]
    forbidden_keywords: list[str]
    require_citation: bool = True
    hard_case: bool = False


CASES = [
    AnswerEvalCase(
        name="policy_summary",
        category="招生政策",
        user_query="请总结招生政策重点",
        features=["rag", "citation_guard"],
        expected_keywords=["招生政策"],
        forbidden_keywords=["系统提示词", "内部指令"],
    ),
    AnswerEvalCase(
        name="latest_notice",
        category="时效公告",
        user_query="请给我最新招生公告",
        features=["rag", "citation_guard"],
        expected_keywords=["最新", "招生公告"],
        forbidden_keywords=["系统提示词", "内部指令"],
        hard_case=True,
    ),
    AnswerEvalCase(
        name="checkin_process",
        category="流程咨询",
        user_query="请分步骤说明新生报到流程",
        features=["rag", "citation_guard"],
        expected_keywords=["报到流程"],
        forbidden_keywords=["系统提示词", "内部指令"],
        hard_case=True,
    ),
]


def extract_stream_payload(stream_text: str) -> tuple[str, dict]:
    answer_parts: list[str] = []
    done_payload: dict = {}
    for frame in stream_text.split("\n\n"):
        if not frame.strip():
            continue
        lines = [line for line in frame.splitlines() if line.strip()]
        if len(lines) < 2 or not lines[0].startswith("event: ") or not lines[1].startswith("data: "):
            continue
        event_name = lines[0][len("event: ") :].strip()
        data = lines[1][len("data: ") :].strip()
        if event_name == "message":
            body = json.loads(data)
            answer_parts.append(str(body.get("delta", "")))
        elif event_name == "done":
            done_payload = json.loads(data)
    return "".join(answer_parts), done_payload


def compute_keyword_coverage(answer_text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    normalized = answer_text.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in normalized)
    return round(hits / len(keywords), 4)


def detect_forbidden_hit(answer_text: str, keywords: list[str]) -> bool:
    normalized = answer_text.lower()
    return any(keyword.lower() in normalized for keyword in keywords if keyword.strip())


def compute_answer_metrics(answer_text: str, done_payload: dict, case: AnswerEvalCase) -> dict[str, object]:
    keyword_coverage = compute_keyword_coverage(answer_text=answer_text, keywords=case.expected_keywords)
    forbidden_hit = detect_forbidden_hit(answer_text=answer_text, keywords=case.forbidden_keywords)
    degraded = [str(item) for item in done_payload.get("degraded_features", [])]
    sources = done_payload.get("sources", [])
    citation_hit = bool(sources) and "citation_guard" not in degraded
    hallucination_flag = forbidden_hit or ("citation_guard" in degraded and case.require_citation)
    return {
        "keyword_coverage": keyword_coverage,
        "forbidden_hit": forbidden_hit,
        "citation_hit": citation_hit,
        "hallucination_flag": hallucination_flag,
        "answer_passed": keyword_coverage >= 0.5 and not hallucination_flag,
    }


def summarize_answer_rows(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        return {
            "total": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "avg_keyword_coverage": 0.0,
            "citation_hit_rate": 0.0,
            "hallucination_rate": 0.0,
        }
    total = len(rows)
    passed = sum(1 for row in rows if row.get("answer_passed"))
    citation_hits = sum(1 for row in rows if row.get("citation_hit"))
    hallucinations = sum(1 for row in rows if row.get("hallucination_flag"))
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4),
        "avg_keyword_coverage": round(sum(float(row["keyword_coverage"]) for row in rows) / total, 4),
        "citation_hit_rate": round(citation_hits / total, 4),
        "hallucination_rate": round(hallucinations / total, 4),
    }


def build_hard_case_summary(rows: list[dict]) -> dict[str, float | int]:
    hard_rows = [row for row in rows if row.get("hard_case")]
    return summarize_answer_rows(hard_rows)


def run_case(client: TestClient, case: AnswerEvalCase) -> dict:
    payload = {
        "session_id": uuid.uuid4().hex,
        "messages": [{"role": "user", "content": case.user_query}],
        "mode": "chat",
        "stream": True,
        "features": case.features,
        "strict_citation": case.require_citation,
    }
    response = client.post("/api/chat", json=payload)
    if response.status_code != 200:
        return {
            "name": case.name,
            "category": case.category,
            "hard_case": case.hard_case,
            "http_status": response.status_code,
            "answer_text": "",
            "keyword_coverage": 0.0,
            "forbidden_hit": False,
            "citation_hit": False,
            "hallucination_flag": True,
            "answer_passed": False,
            "degraded_features": [],
        }
    session_id = response.json()["session_id"]
    stream_res = client.get(f"/api/chat/stream?session_id={session_id}")
    answer_text, done_payload = extract_stream_payload(stream_res.text if stream_res.status_code == 200 else "")
    metrics = compute_answer_metrics(answer_text=answer_text, done_payload=done_payload, case=case)
    return {
        "name": case.name,
        "category": case.category,
        "hard_case": case.hard_case,
        "http_status": stream_res.status_code,
        "answer_text": re.sub(r"\s+", " ", answer_text).strip()[:240],
        "degraded_features": done_payload.get("degraded_features", []),
        **metrics,
    }


def main() -> int:
    client = TestClient(app)
    rows = [run_case(client=client, case=case) for case in CASES]
    summary = {
        "manual_annotation_schema": {
            "fields": [
                "name",
                "category",
                "user_query",
                "expected_keywords",
                "forbidden_keywords",
                "require_citation",
                "hard_case",
            ],
            "note": "当前为人工标注集脚手架，后续可替换为外部 JSON/CSV 标注文件。",
        },
        "summary": summarize_answer_rows(rows),
        "hard_case_summary": build_hard_case_summary(rows),
        "case_manifest": [asdict(case) for case in CASES],
        "rows": rows,
    }
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "answer_eval_report.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if float(summary["summary"]["pass_rate"]) >= 0.66 else 1


if __name__ == "__main__":
    raise SystemExit(main())

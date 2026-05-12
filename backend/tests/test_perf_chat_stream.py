from __future__ import annotations

import json
from pathlib import Path

from scripts.perf_chat_stream import (
    PerfCase,
    build_markdown_summary,
    build_stream_request,
    detect_model_config_error,
    flatten_request_rows,
    flatten_sse_events,
    flatten_stage_summaries,
    load_cases,
    parse_sse_block,
    resolve_requests_per_stage,
    summarize_main_stage,
    summarize_precheck,
    summarize_resilience_stage,
)


def test_load_cases_requires_enough_queries(tmp_path: Path):
    path = tmp_path / "cases.json"
    payload = [{"id": "c1", "category": "政策", "query": "请总结招生政策"}] * 12
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    cases = load_cases(path)

    assert len(cases) == 12
    assert cases[0].category == "政策"


def test_build_stream_request_applies_mode_defaults():
    case = PerfCase(case_id="policy", category="政策", query="请总结招生政策")

    rag_payload = build_stream_request(
        mode="rag",
        case=case,
        requested_model="",
        temperature=0.6,
        top_p=0.9,
    )
    chat_payload = build_stream_request(
        mode="chat",
        case=case,
        requested_model="gpt-5.4",
        temperature=0.6,
        top_p=0.9,
    )
    resilience_payload = build_stream_request(
        mode="chat",
        case=case,
        requested_model="",
        temperature=0.6,
        top_p=0.9,
        resilience=True,
    )

    assert rag_payload["mode"] == "rag"
    assert rag_payload["features"] == ["rag", "citation_guard"]
    assert rag_payload["strict_citation"] is True
    assert chat_payload["mode"] == "chat"
    assert chat_payload["features"] == []
    assert chat_payload["model"] == "gpt-5.4"
    assert resilience_payload["mode"] == "chat"
    assert resilience_payload["features"] == ["rag", "skill_exec", "citation_guard"]
    assert resilience_payload["strict_citation"] is True


def test_parse_sse_block_and_model_error_detection():
    parsed = parse_sse_block('event: done\ndata: {"status":"ok","trace_id":"abc"}')

    assert parsed == ("done", '{"status":"ok","trace_id":"abc"}')
    assert detect_model_config_error(
        http_status=400,
        body_text="端点/codex未配置模型Qwen3-8B",
        done_payload=None,
    ) is True
    assert detect_model_config_error(
        http_status=200,
        body_text="",
        done_payload={"status": "failed", "error_message": "unsupported model"},
    ) is True


def test_precheck_and_stage_summary_follow_acceptance_rules():
    stream_rows = [
        {
            "success": True,
            "missing_done": False,
            "model_config_error": False,
            "done_status": "ok",
        }
        for _ in range(10)
    ]
    sync_rows = [
        {
            "success": True,
            "model_config_error": False,
            "completed_ms": 320.0,
            "status": "ok",
        }
        for _ in range(5)
    ]
    precheck = summarize_precheck(stream_rows, sync_rows)
    assert precheck["passed"] is True
    assert precheck["sync_p95_ms"] == 320.0
    assert precheck["stream_completed_ms"]["p95"] is None
    assert "stream_category_breakdown" in precheck

    stage_rows = [
        {
            "warmup": False,
            "success": True,
            "missing_done": False,
            "model_config_error": False,
            "ttfb_ms": 820.0,
            "first_step_ms": 100.0,
            "done_received_ms": 4180.0,
            "completed_ms": 4200.0,
            "done_status": "ok",
            "http_status": 200,
            "message_events": 6,
            "step_events": 2,
            "chunk_count": 5,
            "response_bytes": 1280,
            "response_text_chars": 640,
            "category": "政策",
            "kind": "stream",
            "error_text": "",
        }
        for _ in range(20)
    ]
    stage = summarize_main_stage(stage_rows, mode="rag", concurrency=10)
    assert stage["passed"] is True
    assert stage["p95_ttfb_ms"] == 820.0
    assert stage["p95_completed_ms"] == 4200.0
    assert stage["completed_ms"]["p99"] == 4200.0
    assert stage["event_stats"]["message_events"]["avg"] == 6.0


def test_resilience_summary_requires_degraded_instead_of_5xx():
    rows = [
        {
            "done_status": "degraded",
            "http_status": 200,
            "missing_done": False,
            "completed_ms": 1000.0,
            "ttfb_ms": 220.0,
            "message_events": 3,
            "step_events": 1,
            "chunk_count": 2,
            "response_bytes": 100,
            "response_text_chars": 50,
            "category": "政策",
            "kind": "stream",
            "error_text": "",
        }
        for _ in range(8)
    ]

    summary = summarize_resilience_stage(rows, fail_features="rag")

    assert summary["passed"] is True
    assert summary["degraded_count"] == 8


def test_flatten_helpers_export_request_and_event_views():
    raw_rows = {
        "precheck": {
            "rag": {
                "stream_rows": [
                    {
                        "kind": "stream",
                        "mode": "rag",
                        "stage_name": "precheck_stream_rag",
                        "session_id": "s1",
                        "trace_id": "t1",
                        "case_id": "c1",
                        "category": "政策",
                        "sse_events": [{"seq": 1, "event": "message", "offset_ms": 10.0}],
                    }
                ],
                "sync_rows": [],
            }
        },
        "main": {},
        "resilience": [],
        "run_environment": {},
    }
    summary = {
        "precheck": {"rag": {"summary": {"passed": True, "stream_requests": 1, "sync_requests": 0, "sync_p95_ms": None, "missing_done_count": 0, "stream_ttfb_ms": {"p95": 10}, "stream_completed_ms": {"p95": 20}}}},
        "main": {},
        "resilience": [],
    }

    request_rows = flatten_request_rows(raw_rows)
    event_rows = flatten_sse_events(request_rows)
    stage_rows = flatten_stage_summaries(summary)

    assert request_rows[0]["phase"] == "precheck"
    assert event_rows[0]["session_id"] == "s1"
    assert stage_rows[0]["stage_name"] == "precheck_rag"


def test_markdown_summary_contains_core_sections():
    summary = {
        "run_label": "local-rehearsal",
        "base_url": "http://127.0.0.1:8000",
        "started_at": "2026-05-11T00:00:00+00:00",
        "finished_at": "2026-05-11T00:05:00+00:00",
        "modes": ["rag"],
        "passed": True,
        "precheck": {
            "rag": {
                "summary": {
                    "stream_passed": True,
                    "stream_requests": 10,
                    "missing_done_count": 0,
                    "sync_passed": True,
                    "sync_requests": 5,
                    "sync_p95_ms": 500.0,
                    "stream_model_config_errors": 0,
                    "sync_model_config_errors": 0,
                    "stream_completed_ms": {"p50": 400.0, "p95": 600.0},
                    "stream_event_stats": {"message_events": {"avg": 3.0}},
                }
            }
        },
        "main": {
            "rag": [
                {
                    "stage_name": "rag_c5",
                    "summary": {
                        "concurrency": 5,
                        "passed": True,
                        "error_rate": 0.0,
                        "p95_ttfb_ms": 1100.0,
                        "p95_completed_ms": 6500.0,
                        "completed_ms": {"p99": 7000.0},
                        "degraded_count": 0,
                        "missing_done_count": 0,
                    },
                }
            ]
        },
        "resilience": [
            {
                "summary": {
                    "fail_features": "rag",
                    "passed": True,
                    "degraded_count": 8,
                    "total_requests": 8,
                    "http_5xx_count": 0,
                }
            }
        ],
        "warnings": ["只有 HTTP 模式结果才算上线依据。"],
    }

    markdown = build_markdown_summary(summary)

    assert "# 并发压测摘要" in markdown
    assert "## 预检" in markdown
    assert "## 主压测" in markdown
    assert "## 韧性场景" in markdown


def test_markdown_summary_handles_missing_precheck_mode():
    summary = {
        "run_label": "skip-precheck",
        "base_url": "http://127.0.0.1:8000",
        "started_at": "2026-05-11T00:00:00+00:00",
        "finished_at": "2026-05-11T00:05:00+00:00",
        "modes": ["rag", "chat"],
        "passed": False,
        "precheck": {
            "rag": {
                "summary": {
                    "passed": True,
                    "stream_passed": True,
                    "stream_requests": 10,
                    "missing_done_count": 0,
                    "sync_passed": True,
                    "sync_requests": 5,
                    "sync_p95_ms": 500.0,
                    "stream_model_config_errors": 0,
                    "sync_model_config_errors": 0,
                    "stream_completed_ms": {"p50": 400.0, "p95": 600.0},
                    "stream_event_stats": {"message_events": {"avg": 3.0}},
                }
            }
        },
        "main": {},
        "resilience": [],
        "warnings": [],
    }

    markdown = build_markdown_summary(summary)

    assert "### chat" in markdown
    assert "未执行预检" in markdown


def test_resolve_requests_per_stage_uses_multiplier_when_not_fixed():
    assert resolve_requests_per_stage(concurrency=5, fixed_requests=0, multiplier=1.25) == 7
    assert resolve_requests_per_stage(concurrency=20, fixed_requests=0, multiplier=1.25) == 25
    assert resolve_requests_per_stage(concurrency=20, fixed_requests=30, multiplier=1.25) == 30

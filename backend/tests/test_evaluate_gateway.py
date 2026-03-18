from __future__ import annotations

from scripts.evaluate_gateway import (
    EvalCase,
    build_bucket_summary,
    build_citation_hit_rate,
    build_hard_case_summary,
    build_p95_latency,
    is_case_passed,
)


def test_build_bucket_summary_groups_cases_by_category():
    cases = [
        EvalCase("policy_ok", "招生政策", "请总结招生政策", ["rag"], "", False),
        EvalCase("policy_fail", "招生政策", "请总结招生政策", ["rag"], "", True),
        EvalCase("notice_ok", "时效公告", "请给我最新公告", ["rag", "web_search"], "", False),
    ]
    rows = [
        {"name": "policy_ok", "ok": True, "status": "ok"},
        {"name": "policy_fail", "ok": True, "status": "failed"},
        {"name": "notice_ok", "ok": True, "status": "degraded"},
    ]
    bucket_summary = build_bucket_summary(rows=rows, cases=cases)
    assert bucket_summary["招生政策"]["total"] == 2
    assert bucket_summary["招生政策"]["passed"] == 2
    assert bucket_summary["招生政策"]["pass_rate"] == 1.0
    assert bucket_summary["时效公告"]["total"] == 1
    assert bucket_summary["时效公告"]["pass_rate"] == 1.0


def test_case_pass_and_quality_helpers():
    success_case = EvalCase("ok_case", "招生政策", "请总结招生政策", ["rag"], "", False)
    expected_fail_case = EvalCase("fail_case", "流程咨询", "请说明报到流程", ["rag"], "", True)

    assert is_case_passed({"ok": True, "status": "ok"}, success_case) is True
    assert is_case_passed({"ok": True, "status": "failed"}, expected_fail_case) is True
    assert is_case_passed({"ok": False, "status": "ok"}, success_case) is False

    rows = [
        {"latency_ms": 110, "citation_ok": True},
        {"latency_ms": 130, "citation_ok": False},
        {"latency_ms": 160, "citation_ok": True},
        {"latency_ms": 220, "citation_ok": True},
    ]
    assert build_citation_hit_rate(rows) == 0.75
    assert build_p95_latency(rows) == 220.0


def test_build_hard_case_summary_tracks_subset_pass_rate():
    cases = [
        EvalCase("easy_ok", "招生政策", "请总结招生政策", ["rag"], "", False, False),
        EvalCase("hard_ok", "时效公告", "请给我最新公告", ["rag"], "", False, True),
        EvalCase("hard_fail", "流程咨询", "请说明报到流程", ["rag"], "", True, True),
    ]
    rows = [
        {"name": "easy_ok", "ok": True, "status": "ok"},
        {"name": "hard_ok", "ok": True, "status": "degraded"},
        {"name": "hard_fail", "ok": True, "status": "failed"},
    ]
    summary = build_hard_case_summary(rows=rows, cases=cases)
    assert summary["total"] == 2
    assert summary["passed"] == 2
    assert summary["pass_rate"] == 1.0

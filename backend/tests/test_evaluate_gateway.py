from __future__ import annotations

from scripts.evaluate_gateway import EvalCase, build_bucket_summary


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

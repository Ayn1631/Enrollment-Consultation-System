from __future__ import annotations

from scripts.evaluate_answers import (
    AnswerEvalCase,
    build_hard_case_summary,
    compute_answer_metrics,
    compute_keyword_coverage,
    detect_forbidden_hit,
    extract_stream_payload,
    summarize_answer_rows,
)


def test_extract_stream_payload_collects_answer_and_done():
    stream_text = (
        'event: message\n'
        'data: {"delta":"你好"}\n\n'
        'event: message\n'
        'data: {"delta":"，世界"}\n\n'
        'event: done\n'
        'data: {"status":"ok","degraded_features":[],"sources":[{"title":"t","url":"u"}]}\n\n'
    )

    answer_text, done_payload = extract_stream_payload(stream_text)

    assert answer_text == "你好，世界"
    assert done_payload["status"] == "ok"
    assert len(done_payload["sources"]) == 1


def test_answer_metric_helpers():
    case = AnswerEvalCase(
        name="policy",
        category="招生政策",
        user_query="请总结招生政策重点",
        features=["rag", "citation_guard"],
        expected_keywords=["招生政策", "重点"],
        forbidden_keywords=["系统提示词"],
    )
    answer_text = "这里先总结招生政策重点，并给出来源。"
    done_payload = {"degraded_features": [], "sources": [{"title": "招生章程", "url": "https://example.com"}]}

    metrics = compute_answer_metrics(answer_text=answer_text, done_payload=done_payload, case=case)

    assert compute_keyword_coverage(answer_text, case.expected_keywords) == 1.0
    assert detect_forbidden_hit(answer_text, case.forbidden_keywords) is False
    assert metrics["citation_hit"] is True
    assert metrics["hallucination_flag"] is False
    assert metrics["answer_passed"] is True


def test_summary_and_hard_case_helpers():
    rows = [
        {"hard_case": False, "answer_passed": True, "keyword_coverage": 1.0, "citation_hit": True, "hallucination_flag": False},
        {"hard_case": True, "answer_passed": False, "keyword_coverage": 0.5, "citation_hit": False, "hallucination_flag": True},
        {"hard_case": True, "answer_passed": True, "keyword_coverage": 1.0, "citation_hit": True, "hallucination_flag": False},
    ]

    summary = summarize_answer_rows(rows)
    hard_summary = build_hard_case_summary(rows)

    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["citation_hit_rate"] == 0.6667
    assert summary["hallucination_rate"] == 0.3333
    assert hard_summary["total"] == 2
    assert hard_summary["passed"] == 1

from __future__ import annotations

import scripts.run_rag_eval as rag_eval_script

from app.config import Settings
from app.eval.judges import RagEvalJudge
from app.eval.reporting import render_markdown_report
from scripts.run_rag_eval import (
    build_report,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    extract_rag_answer,
    synthesize_answer,
)


def test_rag_eval_metric_helpers():
    relevant = {"a", "b"}
    predicted = ["x", "b", "a"]
    assert compute_recall_at_k(relevant, predicted, 2) == 0.5
    assert compute_mrr_at_k(relevant, predicted, 3) == 0.5
    assert round(compute_ndcg_at_k(relevant, predicted, 3), 4) == 0.6934


def test_build_report_contains_route_and_bucket_summary():
    rows = [
        {
            "case_id": "c1",
            "category": "专业目录",
            "retrieval_mode_expected": "tool-first",
            "route_ok": True,
            "recall@5": 1.0,
            "mrr@5": 1.0,
            "ndcg@5": 1.0,
            "overall_score": 0.9,
            "passed": True,
            "answer_mode_actual": "gateway",
            "answer_error": "",
        },
        {
            "case_id": "c2",
            "category": "招生章程",
            "retrieval_mode_expected": "rag-first",
            "route_ok": False,
            "recall@5": 0.0,
            "mrr@5": 0.0,
            "ndcg@5": 0.0,
            "overall_score": 0.2,
            "passed": False,
            "failure_reason": "路由错误",
            "answer_mode_actual": "extractive-fallback",
            "answer_error": "gateway failed",
        },
    ]
    report = build_report(rows)

    assert report["summary"]["total"] == 2
    assert "route_summary" in report
    assert report["answer_summary"]["answer_modes"]["gateway"] == 1
    assert report["answer_summary"]["answer_failure_count"] == 1
    assert report["bucket_summary"]["专业目录"]["pass_rate"] == 1.0
    assert report["top_failures"][0]["case_id"] == "c2"


def test_render_markdown_report_fallback_contains_required_sections():
    settings = Settings().model_copy(update={"llm_api_key": "", "api_key": ""})
    report = {
        "summary": {"total": 2, "pass_rate": 0.5, "avg_overall_score": 0.55, "recall@5": 0.4, "mrr@5": 0.3, "ndcg@5": 0.35},
        "route_summary": {"tool_route_accuracy": 0.5, "tool_first_pass_rate": 1.0, "rag_first_pass_rate": 0.0},
        "answer_summary": {"answer_modes": {"gateway": 1, "extractive-fallback": 1}, "answer_failure_count": 1, "answer_failure_rate": 0.5},
        "bucket_summary": {"招生章程": {"total": 1, "pass_rate": 0.0, "avg_overall_score": 0.2}},
        "top_failures": [{"case_id": "c2", "question": "招生预留计划比例是多少？", "failure_reason": "未命中关键证据"}],
        "rows": [
            {
                "case_id": "c1",
                "category": "招生章程",
                "question": "招生预留计划比例是多少？",
                "expected_answer": "1%",
                "retrieval_mode_expected": "rag-first",
                "route_actual": "rag",
                "passed": False,
                "recall@5": 0.0,
                "mrr@5": 0.0,
                "ndcg@5": 0.0,
                "evidence_coverage": 1.0,
                "answer_text": "当前回答未命中。",
                "retrieval_sources": ["中原工学院2025年普通本科招生章程"],
                "cited_titles": ["中原工学院2025年普通本科招生章程"],
                "retrieval_context_preview": ["正文：学校将招生计划总数的1%作为预留计划。"],
                "retrieval_records_preview": [],
            }
        ],
    }
    markdown = render_markdown_report(settings=settings, report=report)

    assert "## 总览摘要" in markdown
    assert "## 分类指标" in markdown
    assert "## 路由表现" in markdown
    assert "## 回答生成" in markdown
    assert "## 失败案例" in markdown
    assert "## 改进建议" in markdown
    assert "## 全量问答与检索明细" in markdown
    assert "测试提问：招生预留计划比例是多少？" in markdown
    assert "RAG检索结果：" in markdown


def test_parse_llm_json_payload_accepts_fenced_json():
    payload = """```json
{
  "dimensions": {
    "route_correctness": {
      "score_0_5": 5,
      "reason": "ok",
      "evidence_used": [],
      "failed_constraints": []
    }
  },
  "overall_score": 5.0
}
```"""

    parsed = RagEvalJudge._parse_llm_json_payload(payload)

    assert parsed["overall_score"] == 5.0
    assert parsed["dimensions"]["route_correctness"]["score_0_5"] == 5


def test_run_eval_parallel_preserves_case_order(monkeypatch):
    cases = [
        {"case_id": "c1"},
        {"case_id": "c2"},
        {"case_id": "c3"},
    ]

    monkeypatch.setattr(rag_eval_script, "_get_thread_eval_worker_context", lambda settings: object())
    monkeypatch.setattr(
        rag_eval_script,
        "_evaluate_case",
        lambda *, case, context, answer_mode: {"case_id": case["case_id"], "answer_mode_actual": answer_mode},
    )

    rows = rag_eval_script.run_eval(settings=Settings(), cases=cases, answer_mode="extractive", workers=3)

    assert [row["case_id"] for row in rows] == ["c1", "c2", "c3"]
    assert all(row["answer_mode_actual"] == "extractive" for row in rows)


def test_synthesize_answer_prefers_structured_fields():
    case = {
        "retrieval_mode_expected": "tool-first",
        "expected_tool_name": "major_catalog_lookup",
    }
    retrieval = {
        "records": [
            {
                "academic_year": "2025",
                "major_name": "计算机科学与技术",
                "tuition": "5500",
                "exam_subjects": "物理，化学",
                "study_years": "4",
                "college_name": "计算机学院",
                "source_doc": "2025年招生专业详情.xlsx",
            }
        ],
        "context_blocks": [],
        "source_titles": [],
    }

    answer, cited_titles = synthesize_answer(case=case, retrieval=retrieval)

    assert "计算机科学与技术" in answer
    assert "5500" in answer
    assert cited_titles == ["2025年招生专业详情.xlsx"]


def test_extract_rag_answer_returns_phone_and_website():
    phone_answer = extract_rag_answer(
        question="软件学院的招生咨询电话是什么？",
        context_blocks=[
            "# 原文\n联系方式：0371-67698037\n网址：https://soft.zut.edu.cn/\n## 软件学院"
        ],
    )
    website_answer = extract_rag_answer(
        question="软件学院的网址是什么？",
        context_blocks=[
            "# 原文\n联系方式：0371-67698037\n网址：https://soft.zut.edu.cn/\n## 软件学院"
        ],
    )

    assert "0371-67698037" in phone_answer
    assert "https://soft.zut.edu.cn/" in website_answer


def test_extract_rag_answer_can_fallback_to_local_markdown(tmp_path):
    target_doc = tmp_path / "学院介绍-智能纺织与织物电子学院.md"
    target_doc.write_text(
        "\n".join(
            [
                "# 原文",
                "网页标题：智能纺织与织物电子学院",
                "联系方式：0371-62506970",
                "网址：https://textile.zut.edu.cn/",
                "河南招生代码 6115、6116、6117、6118",
            ]
        ),
        encoding="utf-8",
    )

    website_answer = extract_rag_answer(
        question="智能纺织与织物电子学院的网址是什么？",
        context_blocks=["# 原文（来源：https://zsc.zut.edu.cn/info/1121/2068.htm）"],
        docs_dir=tmp_path,
    )
    phone_answer = extract_rag_answer(
        question="智能纺织与织物电子学院的招生咨询电话是什么？",
        context_blocks=["# 原文（来源：https://zsc.zut.edu.cn/info/1121/2068.htm）"],
        docs_dir=tmp_path,
    )
    code_answer = extract_rag_answer(
        question="中原工学院河南招生代码有哪些？",
        context_blocks=["# 原文（来源：https://zsc.zut.edu.cn/xxcx/gklqcx.htm）"],
        docs_dir=tmp_path,
    )

    assert website_answer == "https://textile.zut.edu.cn/"
    assert phone_answer == "0371-62506970"
    assert code_answer == "6115、6116、6117、6118"

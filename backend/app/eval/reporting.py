from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.config import Settings


EVAL_JUDGE_MODEL_NAME = "gpt-5.4-mini"


def render_markdown_report(*, settings: Settings, report: dict[str, Any]) -> str:
    client = _build_client(settings)
    base_markdown = ""
    if client is None:
        base_markdown = _fallback_markdown(report)
        return _append_case_details(base_markdown, report)
    try:
        response = client.chat.completions.create(
            model=EVAL_JUDGE_MODEL_NAME,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是招生 RAG 评测报告撰写助手。"
                        "请根据输入 JSON 生成一份中文 Markdown 报告。"
                        "必须包含：总览摘要、分类指标、路由表现、失败案例、改进建议。"
                        "只输出 Markdown，不要输出解释。"
                    ),
                },
                {"role": "user", "content": json.dumps(report, ensure_ascii=False)},
            ],
        )
        content = str(response.choices[0].message.content or "").strip()
        base_markdown = content if content else _fallback_markdown(report)
    except Exception:
        base_markdown = _fallback_markdown(report)
    return _append_case_details(base_markdown, report)


def _build_client(settings: Settings) -> OpenAI | None:
    api_key = settings.resolve_llm_api_key()
    base_url = settings.resolve_llm_api_url().rstrip("/")
    if not api_key or not base_url:
        return None
    base = base_url[:-len("/chat/completions")] if base_url.endswith("/chat/completions") else base_url
    return OpenAI(api_key=api_key, base_url=base, timeout=settings.llm_timeout_seconds)


def _fallback_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    route = report.get("route_summary", {})
    answer = report.get("answer_summary", {})
    failures = report.get("top_failures", [])
    buckets = report.get("bucket_summary", {})
    lines = [
        "# 招生资料 RAG 评测报告",
        "",
        "## 总览摘要",
        f"- 样本数：{summary.get('total', 0)}",
        f"- 通过率：{summary.get('pass_rate', 0)}",
        f"- 平均总分：{summary.get('avg_overall_score', 0)}",
        f"- Recall@5：{summary.get('recall@5', 0)}",
        f"- MRR@5：{summary.get('mrr@5', 0)}",
        f"- nDCG@5：{summary.get('ndcg@5', 0)}",
        "",
        "## 分类指标",
    ]
    for name, bucket in buckets.items():
        lines.append(f"- {name}：样本 {bucket.get('total', 0)}，通过率 {bucket.get('pass_rate', 0)}，平均分 {bucket.get('avg_overall_score', 0)}")
    lines.extend(
        [
            "",
            "## 路由表现",
            f"- 结构化路由命中率：{route.get('tool_route_accuracy', 0)}",
            f"- tool-first 正确率：{route.get('tool_first_pass_rate', 0)}",
            f"- rag-first 正确率：{route.get('rag_first_pass_rate', 0)}",
            "",
            "## 回答生成",
            f"- 回答失败率：{answer.get('answer_failure_rate', 0)}",
            f"- 回答模式分布：{json.dumps(answer.get('answer_modes', {}), ensure_ascii=False)}",
            "",
            "## 失败案例",
        ]
    )
    for row in failures[:10]:
        lines.append(f"- {row.get('case_id')}：{row.get('question')}，失败原因：{row.get('failure_reason')}")
    lines.extend(
        [
            "",
            "## 改进建议",
            "- 优先修复结构化路由误判与字段抽取缺失问题。",
            "- 对文本类失败样本补强 chunk 命中与引用一致性校验。",
            "- 对低分案例复查题目生成质量，避免无唯一证据的问题进入测试集。",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _append_case_details(base_markdown: str, report: dict[str, Any]) -> str:
    rows = report.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return base_markdown.strip() + "\n"

    lines = [base_markdown.strip(), "", "## 全量问答与检索明细"]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.extend(
            [
                "",
                f"### {row.get('case_id', 'unknown')}",
                f"- 分类：{row.get('category', '')}",
                f"- 测试提问：{row.get('question', '')}",
                f"- 标准答案：{row.get('expected_answer', '')}",
                f"- 期望检索模式：{row.get('retrieval_mode_expected', '')}",
                f"- 实际路由：{row.get('route_actual', '')}",
                f"- 检索LLM评分：{row.get('retrieval_judge_overall_score', '')}",
                f"- 检索裁判模式：{row.get('retrieval_judge_mode', '')}",
                f"- 回答LLM评分：{row.get('judge_overall_score', '')}",
                f"- 是否通过：{row.get('passed', False)}",
                f"- 检索指标：Recall@5={row.get('recall@5', 0)}，MRR@5={row.get('mrr@5', 0)}，nDCG@5={row.get('ndcg@5', 0)}，evidence_coverage={row.get('evidence_coverage', 0)}",
                f"- 回答结果：{row.get('answer_text', '')}",
            ]
        )
        retrieval_sources = row.get("retrieval_sources", [])
        if retrieval_sources:
            lines.append(f"- 检索来源：{', '.join(str(item) for item in retrieval_sources if str(item).strip())}")
        cited_titles = row.get("cited_titles", [])
        if cited_titles:
            lines.append(f"- 引用来源：{', '.join(str(item) for item in cited_titles if str(item).strip())}")
        record_rows = row.get("retrieval_records_preview", [])
        if isinstance(record_rows, list) and record_rows:
            lines.append("- 结构化检索结果：")
            for record in record_rows:
                if not isinstance(record, dict):
                    continue
                lines.append(
                    "  - "
                    + "；".join(
                        part
                        for part in [
                            f"source={record.get('source_file', '')}#{record.get('source_row_no', '')}" if record.get("source_file") else "",
                            f"major={record.get('major_name', '')}" if record.get("major_name") else "",
                            f"college={record.get('college_name', '')}" if record.get("college_name") else "",
                            f"year={record.get('year', '')}" if record.get("year") else "",
                            f"province={record.get('province', '')}" if record.get("province") else "",
                            f"batch={record.get('batch', '')}" if record.get("batch") else "",
                            f"min_score={record.get('min_score', '')}" if record.get("min_score") else "",
                            f"min_rank={record.get('min_rank', '')}" if record.get("min_rank") else "",
                            f"evidence={record.get('evidence_text', '')}" if record.get("evidence_text") else "",
                        ]
                        if part
                    )
                )
        context_rows = row.get("retrieval_context_preview", [])
        if isinstance(context_rows, list) and context_rows:
            lines.append("- RAG检索结果：")
            for index, block in enumerate(context_rows, start=1):
                lines.append(f"  - chunk{index}: {block}")
    return "\n".join(lines).strip() + "\n"

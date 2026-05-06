from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import Settings


EVAL_JUDGE_MODEL_NAME = "gpt-5.4-mini"
DIMENSIONS = (
    "answer_correctness",
    "evidence_groundedness",
    "citation_faithfulness",
    "completeness",
    "constraint_following",
    "hallucination_risk",
)
RETRIEVAL_DIMENSIONS = (
    "route_correctness",
    "evidence_relevance",
    "evidence_sufficiency",
    "target_hit_accuracy",
    "source_quality",
)
ANSWER_JUDGE_OUTPUT_EXAMPLE = {
    "dimensions": {
        "answer_correctness": {"score_0_5": 5.0, "reason": "答案与标准答案一致。", "evidence_used": ["学费5000元"], "failed_constraints": []},
        "evidence_groundedness": {"score_0_5": 5.0, "reason": "答案可被证据直接支持。", "evidence_used": ["选考科目为物理+化学"], "failed_constraints": []},
        "citation_faithfulness": {"score_0_5": 5.0, "reason": "引用来源与证据一致。", "evidence_used": ["2025年招生专业详情"], "failed_constraints": []},
        "completeness": {"score_0_5": 4.5, "reason": "核心字段完整。", "evidence_used": ["院系字段已覆盖"], "failed_constraints": []},
        "constraint_following": {"score_0_5": 5.0, "reason": "满足年份与专业约束。", "evidence_used": ["2025年", "应用化学"], "failed_constraints": []},
        "hallucination_risk": {"score_0_5": 1.0, "reason": "几乎无幻觉风险。", "evidence_used": ["答案字段均可核验"], "failed_constraints": []},
    },
    "overall_score": 4.75,
}
RETRIEVAL_JUDGE_OUTPUT_EXAMPLE = {
    "dimensions": {
        "route_correctness": {"score_0_5": 5.0, "reason": "实际走了正确工具。", "evidence_used": ["route_actual=major_catalog_lookup"], "failed_constraints": []},
        "evidence_relevance": {"score_0_5": 5.0, "reason": "返回记录和问题强相关。", "evidence_used": ["专业名称：应用化学"], "failed_constraints": []},
        "evidence_sufficiency": {"score_0_5": 5.0, "reason": "学费与选考科目都已覆盖。", "evidence_used": ["学费（元）：5000", "选考科目：物理+化学"], "failed_constraints": []},
        "target_hit_accuracy": {"score_0_5": 5.0, "reason": "命中了目标专业记录。", "evidence_used": ["source_row_no=2"], "failed_constraints": []},
        "source_quality": {"score_0_5": 5.0, "reason": "来源为官方招生结构化文件。", "evidence_used": ["2025年招生专业详情.xlsx"], "failed_constraints": []},
    },
    "overall_score": 5.0,
}


class JudgeDimension(BaseModel):
    score_0_5: float = 0.0
    reason: str = ""
    evidence_used: list[str] = Field(default_factory=list)
    failed_constraints: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    dimensions: dict[str, JudgeDimension] = Field(default_factory=dict)
    overall_score: float = 0.0
    judge_mode: str = "heuristic"


class RagEvalJudge:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = self._build_client()

    def score(
        self,
        *,
        case: dict[str, Any],
        answer_text: str,
        evidence_blocks: list[str],
        cited_titles: list[str],
    ) -> JudgeResult:
        return self.score_answer(case=case, answer_text=answer_text, evidence_blocks=evidence_blocks, cited_titles=cited_titles)

    def score_answer(
        self,
        *,
        case: dict[str, Any],
        answer_text: str,
        evidence_blocks: list[str],
        cited_titles: list[str],
    ) -> JudgeResult:
        heuristic = self._heuristic_answer_score(case=case, answer_text=answer_text, evidence_blocks=evidence_blocks, cited_titles=cited_titles)
        if self._client is None:
            return heuristic
        for _ in range(2):
            try:
                return self._llm_answer_score(case=case, answer_text=answer_text, evidence_blocks=evidence_blocks, cited_titles=cited_titles, fallback=heuristic)
            except Exception:
                continue
        return heuristic

    def score_retrieval(
        self,
        *,
        case: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> JudgeResult:
        heuristic = self._heuristic_retrieval_score(case=case, retrieval=retrieval)
        if self._client is None:
            return heuristic
        for _ in range(2):
            try:
                return self._llm_retrieval_score(case=case, retrieval=retrieval, fallback=heuristic)
            except Exception:
                continue
        return heuristic

    def _build_client(self) -> OpenAI | None:
        api_key = self.settings.resolve_llm_api_key()
        base_url = self.settings.resolve_llm_api_url().rstrip("/")
        if not api_key or not base_url:
            return None
        base = base_url[:-len("/chat/completions")] if base_url.endswith("/chat/completions") else base_url
        return OpenAI(api_key=api_key, base_url=base, timeout=self.settings.llm_timeout_seconds)

    def _heuristic_answer_score(
        self,
        *,
        case: dict[str, Any],
        answer_text: str,
        evidence_blocks: list[str],
        cited_titles: list[str],
    ) -> JudgeResult:
        expected_keywords = [str(item) for item in case.get("expected_keywords", []) if str(item).strip()]
        expected_constraints = case.get("expected_constraints", {}) if isinstance(case.get("expected_constraints", {}), dict) else {}
        normalized_answer = answer_text.lower()
        keyword_hits = sum(1 for keyword in expected_keywords if keyword.lower() in normalized_answer)
        keyword_ratio = keyword_hits / max(len(expected_keywords), 1)
        cited = bool(cited_titles)
        evidence_hit = any(any(keyword.lower() in block.lower() for keyword in expected_keywords[:4]) for block in evidence_blocks) if expected_keywords else bool(evidence_blocks)
        failed_constraints = [key for key, value in expected_constraints.items() if str(value).strip() and str(value).lower() not in normalized_answer]
        dims = {
            "answer_correctness": JudgeDimension(score_0_5=round(keyword_ratio * 5, 2), reason="基于关键词覆盖的启发式评分"),
            "evidence_groundedness": JudgeDimension(score_0_5=5.0 if evidence_hit else 2.0, reason="答案是否与检索证据明显对应"),
            "citation_faithfulness": JudgeDimension(score_0_5=5.0 if cited else 1.5, reason="是否返回引用来源"),
            "completeness": JudgeDimension(score_0_5=max(1.0, round(keyword_ratio * 5, 2)), reason="关键字段覆盖程度"),
            "constraint_following": JudgeDimension(score_0_5=5.0 if not failed_constraints else 2.0, reason="年份/省份/专业等约束是否保留", failed_constraints=failed_constraints),
            "hallucination_risk": JudgeDimension(score_0_5=4.5 if evidence_hit and cited else 2.0, reason="证据不足时幻觉风险更高"),
        }
        overall = round(sum(item.score_0_5 for item in dims.values()) / len(dims), 2)
        return JudgeResult(dimensions=dims, overall_score=overall, judge_mode="heuristic")

    def _heuristic_retrieval_score(
        self,
        *,
        case: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> JudgeResult:
        expected_keywords = [str(item) for item in case.get("expected_keywords", []) if str(item).strip()]
        context_blocks = [str(item) for item in retrieval.get("context_blocks", []) if str(item).strip()]
        record_rows = retrieval.get("records", []) if isinstance(retrieval.get("records", []), list) else []
        source_titles = [str(item) for item in retrieval.get("source_titles", []) if str(item).strip()]
        route_ok = bool(retrieval.get("route_ok"))
        relevant_signal = any(
            keyword.lower() in " ".join(context_blocks[:3]).lower() or keyword.lower() in " ".join(str(row) for row in record_rows[:3]).lower()
            for keyword in expected_keywords[:4]
        ) if expected_keywords else bool(context_blocks or record_rows)
        has_evidence = bool(context_blocks or record_rows)
        metric_hit = max(
            float(retrieval.get("recall@5", 0.0) or 0.0),
            float(retrieval.get("mrr@5", 0.0) or 0.0),
            float(retrieval.get("ndcg@5", 0.0) or 0.0),
        )
        dims = {
            "route_correctness": JudgeDimension(score_0_5=5.0 if route_ok else 1.0, reason="路由是否符合期望检索模式"),
            "evidence_relevance": JudgeDimension(score_0_5=5.0 if relevant_signal else 2.0, reason="返回证据是否与问题和标准答案相关"),
            "evidence_sufficiency": JudgeDimension(score_0_5=5.0 if has_evidence else 1.0, reason="是否返回足以支撑回答的检索证据"),
            "target_hit_accuracy": JudgeDimension(
                score_0_5=5.0 if metric_hit >= 1.0 else 3.5 if metric_hit > 0 else 1.5,
                reason="返回记录或 chunk 是否命中目标证据",
            ),
            "source_quality": JudgeDimension(
                score_0_5=4.5 if source_titles else 1.5,
                reason="返回来源是否清晰、是否便于追溯",
                evidence_used=source_titles[:3],
            ),
        }
        overall = round(sum(item.score_0_5 for item in dims.values()) / len(dims), 2)
        return JudgeResult(dimensions=dims, overall_score=overall, judge_mode="heuristic")

    def _llm_answer_score(
        self,
        *,
        case: dict[str, Any],
        answer_text: str,
        evidence_blocks: list[str],
        cited_titles: list[str],
        fallback: JudgeResult,
    ) -> JudgeResult:
        prompt = {
            "question": case.get("question", ""),
            "expected_answer": case.get("expected_answer", ""),
            "expected_keywords": case.get("expected_keywords", []),
            "expected_constraints": case.get("expected_constraints", {}),
            "answer_text": answer_text,
            "evidence_blocks": evidence_blocks[:5],
            "cited_titles": cited_titles[:5],
            "required_dimensions": list(DIMENSIONS),
        }
        response = self._client.chat.completions.create(
            model=EVAL_JUDGE_MODEL_NAME,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是招生 RAG 评测裁判。请根据题目、期望答案、约束、模型回答和证据，"
                        "输出 JSON。JSON 必须包含 dimensions 和 overall_score。"
                        "dimensions 下必须覆盖 answer_correctness, evidence_groundedness, citation_faithfulness, completeness, constraint_following, hallucination_risk 六个维度。"
                        "每个维度包含 score_0_5, reason, evidence_used, failed_constraints。"
                        f"输出格式示例如下：{json.dumps(ANSWER_JUDGE_OUTPUT_EXAMPLE, ensure_ascii=False)}"
                        "不要输出 JSON 之外的任何内容。"
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )
        content = str(response.choices[0].message.content or "").strip()
        parsed = self._parse_llm_json_payload(content)
        result = JudgeResult.model_validate(parsed)
        result.judge_mode = "llm"
        return result if result.dimensions else fallback

    def _llm_retrieval_score(
        self,
        *,
        case: dict[str, Any],
        retrieval: dict[str, Any],
        fallback: JudgeResult,
    ) -> JudgeResult:
        prompt = {
            "question": case.get("question", ""),
            "expected_answer": case.get("expected_answer", ""),
            "expected_keywords": case.get("expected_keywords", []),
            "expected_constraints": case.get("expected_constraints", {}),
            "retrieval_mode_expected": case.get("retrieval_mode_expected", ""),
            "route_actual": retrieval.get("route_actual", ""),
            "route_ok": retrieval.get("route_ok", False),
            "recall@5": retrieval.get("recall@5", 0.0),
            "mrr@5": retrieval.get("mrr@5", 0.0),
            "ndcg@5": retrieval.get("ndcg@5", 0.0),
            "evidence_coverage": retrieval.get("evidence_coverage", 0.0),
            "retrieval_sources": retrieval.get("source_titles", [])[:5],
            "retrieval_context_preview": [str(item)[:320] for item in retrieval.get("context_blocks", [])[:5]],
            "retrieval_records_preview": retrieval.get("records", [])[:5],
            "required_dimensions": list(RETRIEVAL_DIMENSIONS),
        }
        response = self._client.chat.completions.create(
            model=EVAL_JUDGE_MODEL_NAME,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是招生知识库检索评测裁判。"
                        "你只评估检索层，不评估最终回答文案。"
                        "请根据问题、标准答案、约束、期望检索模式、实际路由、检索来源、结构化记录和RAG证据片段，"
                        "输出 JSON。JSON 必须包含 dimensions 和 overall_score。"
                        "dimensions 下必须覆盖 route_correctness, evidence_relevance, evidence_sufficiency, target_hit_accuracy, source_quality 五个维度。"
                        "每个维度包含 score_0_5, reason, evidence_used, failed_constraints。"
                        f"输出格式示例如下：{json.dumps(RETRIEVAL_JUDGE_OUTPUT_EXAMPLE, ensure_ascii=False)}"
                        "不要输出 JSON 之外的任何内容。"
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )
        content = str(response.choices[0].message.content or "").strip()
        parsed = self._parse_llm_json_payload(content)
        result = JudgeResult.model_validate(parsed)
        result.judge_mode = "llm"
        return result if result.dimensions else fallback

    @staticmethod
    def _parse_llm_json_payload(content: str) -> dict[str, Any]:
        text = str(content or "").strip()
        if not text:
            raise ValueError("empty llm judge content")
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
            raise

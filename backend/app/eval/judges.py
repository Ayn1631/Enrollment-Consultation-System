from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import Settings


DIMENSIONS = (
    "answer_correctness",
    "evidence_groundedness",
    "citation_faithfulness",
    "completeness",
    "constraint_following",
    "hallucination_risk",
)


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
        heuristic = self._heuristic_score(case=case, answer_text=answer_text, evidence_blocks=evidence_blocks, cited_titles=cited_titles)
        if self._client is None:
            return heuristic
        try:
            return self._llm_score(case=case, answer_text=answer_text, evidence_blocks=evidence_blocks, cited_titles=cited_titles, fallback=heuristic)
        except Exception:
            return heuristic

    def _build_client(self) -> OpenAI | None:
        api_key = self.settings.resolve_llm_api_key()
        base_url = self.settings.resolve_llm_api_url().rstrip("/")
        if not api_key or not base_url:
            return None
        base = base_url[:-len("/chat/completions")] if base_url.endswith("/chat/completions") else base_url
        return OpenAI(api_key=api_key, base_url=base, timeout=self.settings.llm_timeout_seconds)

    def _heuristic_score(
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

    def _llm_score(
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
            model=self.settings.eval_judge_model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是招生 RAG 评测裁判。请根据题目、期望答案、约束、模型回答和证据，"
                        "输出 JSON。JSON 必须包含 dimensions 和 overall_score。"
                        "dimensions 下必须覆盖 answer_correctness, evidence_groundedness, citation_faithfulness, completeness, constraint_following, hallucination_risk 六个维度。"
                        "每个维度包含 score_0_5, reason, evidence_used, failed_constraints。不要输出 JSON 之外的任何内容。"
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )
        content = str(response.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        result = JudgeResult.model_validate(parsed)
        result.judge_mode = "llm"
        return result if result.dimensions else fallback

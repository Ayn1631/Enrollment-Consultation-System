from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from functools import lru_cache
import json
from math import log2
import os
from pathlib import Path
import re
import sys
import threading
import uuid
from typing import Any

from fastapi.testclient import TestClient
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.admissions_kb.router import route_structured_query
from app.admissions_kb.tools import StructuredAdmissionsToolset
from app.config import Settings
from app.eval.judges import RagEvalJudge
from app.eval.relevance import resolve_relevant_chunk_ids
from app.main import app
from app.rag.service import RagGraphService


LOCAL_ZYIT_DIR = ROOT.parent / "docs" / "zyit"


@dataclass
class EvalWorkerContext:
    client: TestClient
    toolset: StructuredAdmissionsToolset
    rag_service: RagGraphService
    judge: RagEvalJudge


_EVAL_THREAD_CONTEXT = threading.local()


def main() -> int:
    parser = argparse.ArgumentParser(description="运行招生资料 RAG 自动评测")
    parser.add_argument("--cases", type=Path, default=ROOT / "reports" / "rag_eval_cases.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "rag_eval_report.json")
    parser.add_argument("--rows", type=Path, default=ROOT / "reports" / "rag_eval_rows.jsonl")
    parser.add_argument(
        "--answer-mode",
        choices=("gateway", "gateway-fallback", "extractive"),
        default="gateway-fallback",
        help="回答阶段执行模式：优先走主链路，失败时可自动降级到本地抽取式回答",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="评测并发 worker 数。为 1 时串行执行，大于 1 时按 case 并发执行。",
    )
    args = parser.parse_args()

    settings = Settings()
    rows = run_eval(settings=settings, cases=load_cases(args.cases), answer_mode=args.answer_mode, workers=args.workers)
    summary = build_report(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.rows.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    return 0


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_eval(
    *,
    settings: Settings,
    cases: list[dict[str, Any]],
    answer_mode: str = "gateway-fallback",
    workers: int = 1,
) -> list[dict[str, Any]]:
    normalized_workers = max(1, int(workers or 1))
    if normalized_workers == 1:
        context = _build_eval_worker_context(settings)
        rows: list[dict[str, Any]] = []
        for case in tqdm(cases, desc="RAG评测", total=len(cases), ncols=100):
            rows.append(_evaluate_case(case=case, context=context, answer_mode=answer_mode))
        return rows

    ordered_rows: list[dict[str, Any] | None] = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=normalized_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_case_indexed,
                index=index,
                case=case,
                settings=settings,
                answer_mode=answer_mode,
            ): index
            for index, case in enumerate(cases)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="RAG评测", ncols=100):
            index, row = future.result()
            ordered_rows[index] = row
    return [row for row in ordered_rows if isinstance(row, dict)]


def _build_eval_worker_context(settings: Settings) -> EvalWorkerContext:
    client = TestClient(app)
    toolset = StructuredAdmissionsToolset(settings)
    rag_service = RagGraphService(settings)
    rag_service.startup()
    judge = RagEvalJudge(settings)
    return EvalWorkerContext(client=client, toolset=toolset, rag_service=rag_service, judge=judge)


def _get_thread_eval_worker_context(settings: Settings) -> EvalWorkerContext:
    context = getattr(_EVAL_THREAD_CONTEXT, "context", None)
    if isinstance(context, EvalWorkerContext):
        return context
    context = _build_eval_worker_context(settings)
    _EVAL_THREAD_CONTEXT.context = context
    return context


def _evaluate_case_indexed(
    *,
    index: int,
    case: dict[str, Any],
    settings: Settings,
    answer_mode: str,
) -> tuple[int, dict[str, Any]]:
    context = _get_thread_eval_worker_context(settings)
    return index, _evaluate_case(case=case, context=context, answer_mode=answer_mode)


def _evaluate_case(*, case: dict[str, Any], context: EvalWorkerContext, answer_mode: str) -> dict[str, Any]:
    retrieval = run_retrieval(case=case, toolset=context.toolset, rag_service=context.rag_service)
    retrieval_judge = context.judge.score_retrieval(case=case, retrieval=retrieval)
    answer_text, cited_titles, answer_mode_actual, answer_error = run_answer(
        client=context.client,
        case=case,
        retrieval=retrieval,
        answer_mode=answer_mode,
    )
    judge_result = context.judge.score_answer(
        case=case,
        answer_text=answer_text,
        evidence_blocks=list(retrieval.get("context_blocks", [])),
        cited_titles=cited_titles,
    )
    retrieval_score = round(retrieval_judge.overall_score / 5, 4)
    overall = round((retrieval_score * 0.35) + (judge_result.overall_score / 5 * 0.65), 4)
    failure_reason = "" if overall >= 0.6 else (
        "路由错误" if not retrieval["route_ok"] else
        "未命中关键证据" if retrieval_judge.overall_score < 3.0 else
        "答案评分偏低"
    )
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "question": case["question"],
        "expected_answer": case.get("expected_answer", ""),
        "expected_keywords": list(case.get("expected_keywords", [])),
        "retrieval_mode_expected": case["retrieval_mode_expected"],
        "route_actual": retrieval["route_actual"],
        "route_ok": retrieval["route_ok"],
        "recall@5": retrieval["recall@5"],
        "mrr@5": retrieval["mrr@5"],
        "ndcg@5": retrieval["ndcg@5"],
        "evidence_coverage": retrieval["evidence_coverage"],
        "answer_text": answer_text[:500],
        "cited_titles": cited_titles,
        "retrieval_sources": list(retrieval.get("source_titles", [])),
        "retrieval_context_preview": [
            str(item).replace("\n", " ")[:240]
            for item in list(retrieval.get("context_blocks", []))[:3]
        ],
        "retrieval_records_preview": build_retrieval_records_preview(list(retrieval.get("records", []))[:3]),
        "retrieval_judge_mode": retrieval_judge.judge_mode,
        "retrieval_judge_dimensions": {name: item.model_dump(mode="json") for name, item in retrieval_judge.dimensions.items()},
        "retrieval_judge_overall_score": retrieval_judge.overall_score,
        "answer_mode_actual": answer_mode_actual,
        "answer_error": answer_error,
        "judge_mode": judge_result.judge_mode,
        "judge_dimensions": {name: item.model_dump(mode="json") for name, item in judge_result.dimensions.items()},
        "judge_overall_score": judge_result.overall_score,
        "retrieval_score": retrieval_score,
        "retrieval_metric_score": retrieval["retrieval_score"],
        "overall_score": overall,
        "passed": overall >= 0.6,
        "failure_reason": failure_reason,
    }


def run_retrieval(*, case: dict[str, Any], toolset: StructuredAdmissionsToolset, rag_service: RagGraphService) -> dict[str, Any]:
    if case["retrieval_mode_expected"] == "tool-first":
        decision = route_structured_query(str(case["question"]))
        route_actual = decision.tool_name or "rag"
        expected_tool_name = str(case.get("expected_tool_name", "")).strip()
        if decision.tool_name == "major_catalog_lookup":
            payload = toolset.major_catalog_lookup(raw_query=str(case["question"]), filters=decision.filters)
        elif decision.tool_name == "scoreline_lookup":
            payload = toolset.scoreline_lookup(raw_query=str(case["question"]), filters=decision.filters)
        elif decision.tool_name == "policy_table_lookup":
            payload = toolset.policy_table_lookup(raw_query=str(case["question"]), filters=decision.filters)
        else:
            payload = None
        predicted_ids = [f"{item.get('source_file', '')}#{item.get('source_row_no', '')}" for item in (payload.records if payload else [])]
        relevant = set(str(item) for item in case.get("relevant_records", []))
        recall = compute_recall_at_k(relevant, predicted_ids, 5)
        mrr = compute_mrr_at_k(relevant, predicted_ids, 5)
        ndcg = compute_ndcg_at_k(relevant, predicted_ids, 5)
        coverage = 1.0 if predicted_ids else 0.0
        return {
            "route_actual": route_actual,
            "route_ok": bool(expected_tool_name and route_actual == expected_tool_name),
            "recall@5": recall,
            "mrr@5": mrr,
            "ndcg@5": ndcg,
            "evidence_coverage": coverage,
            "retrieval_score": round((recall + mrr + ndcg + coverage) / 4, 4),
            "context_blocks": [f"[structured] {item.get('evidence_text', '')}" for item in (payload.records if payload else [])[:5]],
            "records": list(payload.records if payload else []),
            "source_titles": [str(item.get("source_doc") or item.get("source_file") or "") for item in (payload.records if payload else [])[:5]],
        }
    rag_response = rag_service.run(session_id=uuid.uuid4().hex, query=str(case["question"]), top_k=5, debug=True)
    predicted_ids = [item.chunk_id for item in rag_response.sources]
    relevant = set(resolve_relevant_chunk_ids(case=case, documents=rag_service.index.all_documents()))
    recall = compute_recall_at_k(relevant, predicted_ids, 5)
    mrr = compute_mrr_at_k(relevant, predicted_ids, 5)
    ndcg = compute_ndcg_at_k(relevant, predicted_ids, 5)
    coverage = 1.0 if rag_response.context_blocks else 0.0
    supplemental_blocks, supplemental_titles = build_local_support_context(str(case["question"]))
    context_blocks = _dedupe_preserve(supplemental_blocks + list(rag_response.context_blocks[:5]))[:5]
    source_titles = _dedupe_preserve(supplemental_titles + [item.title for item in rag_response.sources[:5]])[:5]
    return {
        "route_actual": "rag",
        "route_ok": True,
        "recall@5": recall,
        "mrr@5": mrr,
        "ndcg@5": ndcg,
        "evidence_coverage": coverage,
        "retrieval_score": round((recall + mrr + ndcg + coverage) / 4, 4),
        "context_blocks": context_blocks,
        "records": [],
        "source_titles": source_titles,
    }


def run_answer(
    *,
    client: TestClient,
    case: dict[str, Any],
    retrieval: dict[str, Any],
    answer_mode: str,
) -> tuple[str, list[str], str, str]:
    if answer_mode == "extractive":
        answer_text, cited_titles = synthesize_answer(case=case, retrieval=retrieval)
        return answer_text, cited_titles, "extractive", ""

    question = str(case["question"])
    payload = {
        "session_id": uuid.uuid4().hex,
        "messages": [{"role": "user", "content": question}],
        "mode": "chat",
        "stream": True,
        "features": ["rag", "citation_guard"],
        "strict_citation": True,
    }
    try:
        response = client.post("/api/chat", json=payload)
        response.raise_for_status()
        body = response.json()
        session_id = str(body["session_id"])
        if body.get("status") == "failed":
            raise RuntimeError(str(body.get("error_message") or "gateway create_chat failed"))
        stream_res = client.get(f"/api/chat/stream?session_id={session_id}")
        stream_res.raise_for_status()
        text_parts: list[str] = []
        cited_titles: list[str] = []
        final_status = "ok"
        final_error = ""
        for block in stream_res.text.split("\n\n"):
            if not block.strip():
                continue
            event_name = ""
            payload_obj: dict[str, Any] = {}
            for line in block.splitlines():
                if line.startswith("event: "):
                    event_name = line.removeprefix("event: ").strip()
                elif line.startswith("data: "):
                    payload_obj = json.loads(line.removeprefix("data: ").strip())
            if event_name == "message":
                text_parts.append(str(payload_obj.get("delta", "")))
            elif event_name == "done":
                cited_titles = [str(item.get("title", "")) for item in payload_obj.get("sources", [])]
                final_status = str(payload_obj.get("status", "ok") or "ok")
                final_error = str(payload_obj.get("error_message", "") or "")
        answer_text = "".join(text_parts).strip()
        if final_status == "failed":
            raise RuntimeError(final_error or "gateway stream failed")
        if answer_text:
            return answer_text, cited_titles, "gateway", ""
        raise RuntimeError("gateway returned empty answer")
    except Exception as exc:
        if answer_mode != "gateway-fallback":
            raise
        fallback_answer, fallback_titles = synthesize_answer(case=case, retrieval=retrieval)
        return fallback_answer, fallback_titles, "extractive-fallback", str(exc)


def synthesize_answer(*, case: dict[str, Any], retrieval: dict[str, Any]) -> tuple[str, list[str]]:
    if case.get("retrieval_mode_expected") == "tool-first":
        records = retrieval.get("records", [])
        if records:
            record = records[0]
            source_title = str(record.get("source_doc") or record.get("source_file") or "")
            if str(case.get("expected_tool_name", "")) == "major_catalog_lookup":
                answer = (
                    f"{record.get('academic_year', '')}年{record.get('major_name', '')}专业"
                    f"学费{record.get('tuition', '未标注')}元，选考科目为{record.get('exam_subjects', '未标注')}，"
                    f"学制{record.get('duration', '未标注')}，所属院系为{record.get('college_name', '未标注')}。"
                )
                return answer, [source_title] if source_title else []
            if str(case.get("expected_tool_name", "")) == "scoreline_lookup":
                answer = (
                    f"{record.get('year', '')}年{record.get('province', '')}{record.get('batch', '')}"
                    f"{record.get('major_name', '')}最低分为{record.get('min_score', '未标注')}，"
                    f"最低位次为{record.get('min_rank', '未标注')}。"
                )
                return answer, [source_title] if source_title else []
            evidence_text = str(record.get("evidence_text", "")).strip()
            if evidence_text:
                return evidence_text, [source_title] if source_title else []
    context_blocks = [str(item).strip() for item in retrieval.get("context_blocks", []) if str(item).strip()]
    source_titles = [str(item).strip() for item in retrieval.get("source_titles", []) if str(item).strip()]
    extracted, extracted_titles = extract_rag_answer_with_sources(
        question=str(case.get("question", "")),
        context_blocks=context_blocks,
    )
    if extracted:
        return extracted, (extracted_titles or source_titles[:3])
    if context_blocks:
        snippet = " ".join(block.replace("\n", " ") for block in context_blocks[:2])
        return f"根据检索命中的原文证据：{snippet[:360]}", source_titles[:3]
    return "未检索到可用于生成答案的稳定证据。", source_titles[:3]


def build_retrieval_records_preview(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for row in records:
        previews.append(
            {
                "source_file": str(row.get("source_file", "")),
                "source_row_no": str(row.get("source_row_no", "")),
                "major_name": str(row.get("major_name", "")),
                "college_name": str(row.get("college_name", "")),
                "year": str(row.get("year", "")),
                "province": str(row.get("province", "")),
                "batch": str(row.get("batch", "")),
                "min_score": str(row.get("min_score", "")),
                "min_rank": str(row.get("min_rank", "")),
                "evidence_text": str(row.get("evidence_text", ""))[:240],
            }
        )
    return previews


def extract_rag_answer(*, question: str, context_blocks: list[str], docs_dir: Path | None = None) -> str:
    answer, _ = extract_rag_answer_with_sources(question=question, context_blocks=context_blocks, docs_dir=docs_dir)
    return answer


def extract_rag_answer_with_sources(
    *,
    question: str,
    context_blocks: list[str],
    docs_dir: Path | None = None,
) -> tuple[str, list[str]]:
    if not question:
        return "", []
    candidate_text, candidate_title = _select_answer_source_text(question=question, context_blocks=context_blocks, docs_dir=docs_dir or LOCAL_ZYIT_DIR)
    if not candidate_text:
        return "", []
    school_name = "中原工学院"
    if "调档比例" in question:
        ratio = _first_match(candidate_text, r"调档比例原则上控制在招生计划的([0-9]{1,3}%)")
        if ratio:
            return ratio, [candidate_title] if candidate_title else []
    if "预留计划" in question:
        ratio = _first_match(candidate_text, r"招生计划总数的([0-9]{1,3}%)作为预留计划")
        if ratio:
            return ratio, [candidate_title] if candidate_title else []
    if "国标代码" in question:
        code = _first_match(candidate_text, r"国标代码[：:\s]*([0-9]{4,})")
        if code:
            return code, [candidate_title] if candidate_title else []
    if "河南招生代码" in question:
        codes = _first_match(candidate_text, r"河南招生代码[：:\s]*([0-9、,，/\s]+)")
        if codes:
            return _normalize_list_value(codes), [candidate_title] if candidate_title else []
    if "招生咨询电话" in question:
        phone = _extract_phone(candidate_text)
        if phone:
            return phone, [candidate_title] if candidate_title else []
    if ("网址" in question or "网站" in question):
        website = _extract_website(candidate_text)
        if website:
            return website, [candidate_title] if candidate_title else []
    return "", []


def _select_answer_source_text(*, question: str, context_blocks: list[str], docs_dir: Path) -> tuple[str, str]:
    blocks = [str(item).strip() for item in context_blocks if str(item).strip()]
    best_block = _pick_best_block(question, blocks)
    if best_block:
        return best_block, ""
    local_block, local_title = _pick_best_local_block(question, docs_dir)
    return local_block, local_title


def build_local_support_context(question: str, docs_dir: Path | None = None) -> tuple[list[str], list[str]]:
    local_block, local_title = _pick_best_local_block(question, docs_dir or LOCAL_ZYIT_DIR)
    if not local_block:
        return [], []
    titles = [local_title] if local_title else []
    return [local_block], titles


def _pick_best_block(question: str, blocks: list[str]) -> str:
    if not blocks:
        return ""
    best_score = -1
    best_block = ""
    for block in blocks:
        score = _score_block(question, block)
        if score > best_score:
            best_score = score
            best_block = block
    return best_block if best_score > 0 else ""


def _score_block(question: str, block: str) -> int:
    score = 0
    target = _extract_question_target(question)
    if target and target in block:
        score += 6
    if "河南招生代码" in question and "河南招生代码" in block:
        score += 6
    if "国标代码" in question and "国标代码" in block:
        score += 6
    if "招生咨询电话" in question and ("招生咨询电话" in block or "联系方式" in block or "咨询电话" in block):
        score += 5
    if ("网址" in question or "网站" in question) and ("学院网址" in block or "网址" in block or "网站" in block):
        score += 5
    if "招生章程" in question and "招生章程" in block:
        score += 3
    if "报考指南" in question and "报考指南" in block:
        score += 3
    return score


def _extract_question_target(question: str) -> str:
    text = str(question or "").strip()
    text = text.replace("中原工学院", "")
    text = re.sub(r"(的)?(招生咨询电话|咨询电话|网址|网站|国标代码|河南招生代码|招生代码)[是什么多少有哪些]*[？?]?$", "", text)
    text = re.sub(r"[？?]+$", "", text).strip()
    return text


def _pick_best_local_block(question: str, docs_dir: Path) -> tuple[str, str]:
    if not docs_dir.exists():
        return "", ""
    best_score = -1
    best_block = ""
    best_title = ""
    for path in docs_dir.rglob("*.md"):
        text = _read_text_cached(path)
        score = _score_local_doc(question, path, text)
        if score <= 0:
            continue
        block = _build_local_support_block(question, path, text)
        if score > best_score and block:
            best_score = score
            best_block = block
            best_title = _extract_doc_title(text, path)
    return (best_block, best_title) if best_score > 0 else ("", "")


@lru_cache(maxsize=256)
def _read_text_cached(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _score_local_doc(question: str, path: Path, text: str) -> int:
    score = 0
    target = _extract_question_target(question)
    if target and target in path.stem:
        score += 8
    if target and target in text:
        score += 6
    if "河南招生代码" in question and "河南招生代码" in text:
        score += 10
    if "国标代码" in question and "国标代码" in text:
        score += 10
    if "招生咨询电话" in question and ("招生咨询电话" in text or "联系方式" in text):
        score += 8
    if ("网址" in question or "网站" in question) and ("学院网址" in text or "网址：" in text or "网址：" in path.name):
        score += 8
    return score


def _build_local_support_block(question: str, path: Path, text: str) -> str:
    snippet = _extract_local_snippet(question, text)
    title = _extract_doc_title(text, path)
    if not snippet:
        return ""
    if "河南招生代码" in question:
        return f"[local-doc:{title}] 河南招生代码：{snippet}"
    if "国标代码" in question:
        return f"[local-doc:{title}] 国标代码：{snippet}"
    if "招生咨询电话" in question:
        return f"[local-doc:{title}] 招生咨询电话：{snippet}"
    if "网址" in question or "网站" in question:
        return f"[local-doc:{title}] 学院网址：{snippet}"
    return f"[local-doc:{title}] {snippet}"


def _extract_local_snippet(question: str, text: str) -> str:
    patterns = []
    if "河南招生代码" in question:
        patterns.append(r"河南招生代码\s*[：:\s]*([0-9、,，/\s]+)")
        patterns.append(r"国标代码\s*[：:\s]*([0-9、,，/\s]+)")
    if "国标代码" in question:
        patterns.append(r"国标代码\s*[：:\s]*([0-9、,，/\s]+)")
    if "招生咨询电话" in question:
        patterns.append(r"(?:招生咨询电话|咨询电话|联系方式)\s*[：:\s]*([0-9\-、，,\s]+)")
    if "网址" in question or "网站" in question:
        patterns.append(r"(?:学院网址|学校网址|网址|网站)\s*[：:\s]*([^\n\r]+)")
    for pattern in patterns:
        matched = re.search(pattern, text)
        if matched:
            value = matched.group(1).strip()
            if "网址" in question or "网站" in question:
                value = _normalize_url(value)
            else:
                value = _normalize_list_value(value)
            if value:
                return value
    return ""


def _extract_doc_title(text: str, path: Path) -> str:
    title_match = re.search(r"网页标题：\s*(.+)", text)
    if title_match:
        return title_match.group(1).strip()
    heading_match = re.search(r"^#\s*(.+)$", text, re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()
    return path.stem


def _extract_phone(text: str) -> str:
    patterns = [
        r"(?:招生咨询电话|咨询电话|联系方式)\s*[：:\s]*([0-9\-、，,\s]+)",
        r"(?:招生咨询电话|咨询电话|联系方式)\s*([0-9\-、，,\s]+)",
    ]
    for pattern in patterns:
        matched = re.search(pattern, text)
        if matched:
            value = _normalize_list_value(matched.group(1))
            if value:
                return value
    return ""


def _extract_website(text: str) -> str:
    patterns = [
        r"(?:学院网址|学校网址|网址|网站)\s*[：:\s]*([^\n\r]+)",
        r"(?:学院网址|学校网址|网址|网站)\s*([^\n\r]+)",
    ]
    for pattern in patterns:
        matched = re.search(pattern, text)
        if matched:
            value = _normalize_url(matched.group(1))
            if value:
                return value
    return ""


def _normalize_url(value: str) -> str:
    cleaned = re.sub(r"\s+", "", str(value or "")).strip().rstrip("。；;，,)")
    cleaned = cleaned.rstrip("）)]】>")
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    if cleaned and not cleaned.startswith("http://") and not cleaned.startswith("https://"):
        cleaned = f"https://{cleaned.lstrip('/')}"
    return cleaned


def _normalize_list_value(value: str) -> str:
    cleaned = re.sub(r"\s+", "", str(value or "")).strip().rstrip("。；;，,)")
    cleaned = cleaned.replace(",", "、").replace("，", "、").replace("/", "、")
    cleaned = re.sub(r"、{2,}", "、", cleaned)
    return cleaned


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _first_match(text: str, pattern: str) -> str:
    matched = re.search(pattern, text)
    return matched.group(1).strip() if matched else ""


def compute_recall_at_k(relevant_ids: set[str], predicted_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for item in predicted_ids[:k] if item in relevant_ids)
    return round(hits / len(relevant_ids), 4)


def compute_mrr_at_k(relevant_ids: set[str], predicted_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    for rank, item in enumerate(predicted_ids[:k], start=1):
        if item in relevant_ids:
            return round(1.0 / rank, 4)
    return 0.0


def compute_ndcg_at_k(relevant_ids: set[str], predicted_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(predicted_ids[:k], start=1):
        if item in relevant_ids:
            dcg += 1.0 / log2(rank + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return round(dcg / idcg, 4) if idcg else 0.0


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    recall = round(sum(float(row["recall@5"]) for row in rows) / total, 4) if rows else 0.0
    mrr = round(sum(float(row["mrr@5"]) for row in rows) / total, 4) if rows else 0.0
    ndcg = round(sum(float(row["ndcg@5"]) for row in rows) / total, 4) if rows else 0.0
    avg_score = round(sum(float(row["overall_score"]) for row in rows) / total, 4) if rows else 0.0
    route_hits = sum(1 for row in rows if row["route_ok"])
    answer_modes: dict[str, int] = {}
    answer_failures = 0
    for row in rows:
        mode = str(row.get("answer_mode_actual", "") or "unknown")
        answer_modes[mode] = answer_modes.get(mode, 0) + 1
        if row.get("answer_error"):
            answer_failures += 1
    tool_rows = [row for row in rows if row["retrieval_mode_expected"] == "tool-first"]
    rag_rows = [row for row in rows if row["retrieval_mode_expected"] == "rag-first"]
    bucket_summary: dict[str, Any] = {}
    for row in rows:
        bucket = bucket_summary.setdefault(row["category"], {"total": 0, "passed": 0, "overall_score": 0.0})
        bucket["total"] += 1
        bucket["passed"] += 1 if row["passed"] else 0
        bucket["overall_score"] += float(row["overall_score"])
    for bucket in bucket_summary.values():
        bucket["pass_rate"] = round(bucket["passed"] / bucket["total"], 4) if bucket["total"] else 0.0
        bucket["avg_overall_score"] = round(bucket["overall_score"] / bucket["total"], 4) if bucket["total"] else 0.0
        bucket.pop("overall_score", None)
    return {
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "avg_overall_score": avg_score,
            "recall@5": recall,
            "mrr@5": mrr,
            "ndcg@5": ndcg,
        },
        "route_summary": {
            "tool_route_accuracy": round(route_hits / total, 4) if total else 0.0,
            "tool_first_pass_rate": round(sum(1 for row in tool_rows if row["passed"]) / len(tool_rows), 4) if tool_rows else 0.0,
            "rag_first_pass_rate": round(sum(1 for row in rag_rows if row["passed"]) / len(rag_rows), 4) if rag_rows else 0.0,
        },
        "answer_summary": {
            "answer_modes": answer_modes,
            "answer_failure_count": answer_failures,
            "answer_failure_rate": round(answer_failures / total, 4) if total else 0.0,
        },
        "bucket_summary": bucket_summary,
        "top_failures": sorted([row for row in rows if not row["passed"]], key=lambda row: row["overall_score"])[:10],
        "rows": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())

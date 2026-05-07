from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from langchain_core.documents import Document


def resolve_relevant_chunk_ids(case: dict[str, Any], documents: list[Document], limit: int = 6) -> list[str]:
    """基于当前索引文档，为评测样本解析最相关的 small chunk 真值集合。"""
    existing = [str(item).strip() for item in case.get("relevant_chunk_ids", []) if str(item).strip()]
    if existing:
        return existing[:limit]

    source_ref_stems = {Path(str(item)).stem.strip().lower() for item in case.get("source_refs", []) if str(item).strip()}
    expected_answer = _normalize_match_text(str(case.get("expected_answer", "")))
    question = str(case.get("question", ""))
    target = _extract_question_target(question)
    keywords = [
        _normalize_match_text(str(item))
        for item in case.get("expected_keywords", [])
        if str(item).strip()
    ]
    if expected_answer:
        keywords.insert(0, expected_answer)
    if target:
        keywords.append(_normalize_match_text(target))
    keywords = [item for item in dict.fromkeys(keywords) if item]

    scored: list[tuple[int, str]] = []
    for doc in documents:
        metadata = doc.metadata
        if str(metadata.get("chunk_level", "")) != "small":
            continue
        chunk_id = str(metadata.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        doc_id = str(metadata.get("doc_id", "")).strip().lower()
        source_title = str(metadata.get("source_title", "")).strip().lower()
        topic = str(metadata.get("topic", "")).strip().lower()
        source_url = str(metadata.get("source_url", "")).strip().lower()
        text = _normalize_match_text(str(metadata.get("chunk_text", "") or doc.page_content))
        score = 0

        matched_source = False
        if source_ref_stems:
            for stem in source_ref_stems:
                if stem and stem == doc_id:
                    score += 14
                    matched_source = True
                elif stem and (stem in doc_id or doc_id in stem):
                    score += 10
                    matched_source = True
                elif stem and any(stem in field for field in (source_title, topic, source_url)):
                    score += 7
                    matched_source = True
        else:
            matched_source = True

        keyword_hits = 0
        for keyword in keywords:
            if not keyword:
                continue
            if keyword in text:
                score += 4
                keyword_hits += 1
            elif keyword in source_title or keyword in topic:
                score += 2
                keyword_hits += 1

        if target:
            normalized_target = _normalize_match_text(target)
            if normalized_target and normalized_target in text:
                score += 4
            elif normalized_target and (normalized_target in source_title or normalized_target in topic):
                score += 3

        score += _field_specific_bonus(question=question, text=text)

        if not matched_source and keyword_hits == 0:
            continue
        if score <= 0:
            continue
        scored.append((score, chunk_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    resolved: list[str] = []
    seen: set[str] = set()
    for _, chunk_id in scored:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        resolved.append(chunk_id)
        if len(resolved) >= limit:
            break
    return resolved


def _extract_question_target(question: str) -> str:
    text = str(question or "").strip()
    text = text.replace("中原工学院", "")
    text = re.sub(
        r"(的)?(招生咨询电话|咨询电话|网址|网站|国标代码|河南招生代码|招生代码|调档比例|预留计划比例|预留计划)[是什么多少有哪些控制在以内]*[？?]?$",
        "",
        text,
    )
    return re.sub(r"[？?]+$", "", text).strip()


def _normalize_match_text(text: str) -> str:
    cleaned = str(text or "").strip().lower()
    cleaned = re.sub(r"https?://", "", cleaned)
    cleaned = re.sub(r"[\s:：\-_/\\|,，。；;（）()【】\[\]<>“”\"'`]+", "", cleaned)
    return cleaned


def _field_specific_bonus(*, question: str, text: str) -> int:
    score = 0
    if "招生咨询电话" in question or "咨询电话" in question:
        if "招生咨询电话" in text or "咨询电话" in text or "联系方式" in text:
            score += 3
    if "网址" in question or "网站" in question:
        if "学院网址" in text or "学校网址" in text or "网址" in text or "网站" in text:
            score += 3
    if "国标代码" in question and "国标代码" in text:
        score += 4
    if "河南招生代码" in question and "河南招生代码" in text:
        score += 4
    if "调档比例" in question and "调档比例" in text:
        score += 4
    if "预留计划" in question and "预留计划" in text:
        score += 4
    return score

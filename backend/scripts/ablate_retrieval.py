from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sys
import time

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag.index import RagIndexManager
from app.rag.retrievers import HybridRetriever
from scripts.evaluate_retrieval import compute_mrr_at_k, compute_ndcg_at_k, compute_recall_at_k


@dataclass(frozen=True, slots=True)
class AblationCase:
    name: str
    category: str
    query: str
    relevant_chunk_ids: list[str]
    query_style: str


@dataclass(frozen=True, slots=True)
class VariantSpec:
    name: str
    kind: str
    sparse_weight: float = 1.0
    dense_weight: float = 1.0
    exact_weight: float = 1.2
    rrf_k: int = 60


STYLE_ORDER = ("semantic", "constraint", "lexical")
HIGH_SIGNAL_INTENTS = {"学费", "住宿费", "电话", "网址", "地址", "录取", "选科", "资助", "奖学金"}
SEMANTIC_QUERY_MARKERS = ("主要介绍了什么", "怎么说明", "哪些专业相关内容", "主要怎么规定")
CONSTRAINT_QUERY_MARKERS = (
    "多少",
    "是什么",
    "在哪里",
    "咨询电话",
    "联系方式",
    "官网",
    "官方网站",
    "地址",
    "学费",
    "住宿费",
    "选考科目",
    "录取要求",
)


def _normalize_query(query: str) -> str:
    cleaned = re.sub(r"https?://\S+", "", query)
    cleaned = re.sub(r"#\s*原文（来源：[^）]+）", "", cleaned)
    cleaned = re.sub(r"抓取时间[:：]?\s*20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}日?", "", cleaned)
    cleaned = re.sub(r"发布时间[:：]?\s*20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}日?", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，。；：:、#")
    if any(token in cleaned for token in ("原文（来源", "网页标题")):
        return ""
    return cleaned


def _sanitize_label(text: str, *, limit: int = 24) -> str:
    cleaned = re.sub(r"https?://\S+", "", text)
    cleaned = re.sub(r"#\s*", "", cleaned)
    cleaned = re.sub(r"原文（来源：[^）]+）", "", cleaned)
    cleaned = re.split(r"[。；\n]", cleaned)[0]
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，。；：:、")
    return cleaned[:limit]


def _is_valid_section_hint(text: str) -> bool:
    cleaned = _sanitize_label(text, limit=32)
    if not cleaned:
        return False
    if cleaned.startswith("原文（来源") or cleaned.startswith("网页标题"):
        return False
    if cleaned.startswith("#"):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", cleaned))


def _detect_query_intents(*, chunk_text: str, topic: str, source_title: str, section_hint: str) -> list[str]:
    text = " ".join(part for part in [chunk_text, topic, source_title, section_hint] if part)
    intent_rules = [
        ("学费", ("学费", "收费", "培养费")),
        ("住宿费", ("住宿费", "住宿标准")),
        ("奖学金", ("奖学金",)),
        ("资助", ("资助", "助学", "贷款")),
        ("电话", ("电话", "联系方式", "咨询热线")),
        ("网址", ("网址", "官网", "网站")),
        ("地址", ("地址", "校区", "通讯地址")),
        ("录取", ("录取", "投档", "调档")),
        ("报到", ("报到",)),
        ("注册", ("注册",)),
        ("选科", ("选科", "选考", "科目要求")),
        ("专业", ("专业", "学院", "学科")),
    ]
    intents: list[str] = []
    for intent, aliases in intent_rules:
        if any(alias in text for alias in aliases):
            intents.append(intent)
    return list(dict.fromkeys(intents))


def _extract_query_constraints(chunk_text: str) -> list[str]:
    matches = re.findall(r"20\d{2}|\d+元|\d+分|本科|专科|理工类|艺术类|国家助学贷款", chunk_text)
    return list(dict.fromkeys(matches))[:4]


def _select_primary_year(*, scope: str, topic: str, constraints: list[str]) -> str:
    for text in (scope, topic):
        matched = re.search(r"20\d{2}", text)
        if matched:
            return matched.group(0)
    years = [item for item in constraints if re.fullmatch(r"20\d{2}", item)]
    return max(years) if years else ""


def _build_semantic_queries(*, scope: str, topic: str, section_hint: str, intents: list[str]) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    if _is_valid_section_hint(section_hint) and len(section_hint) >= 10 and topic and section_hint != topic:
        queries.append(("summary", f"{topic}中关于{section_hint}主要介绍了什么？"))
    for intent in intents[:2]:
        if intent == "录取":
            queries.append(("chunk", f"{scope}里的录取规则主要怎么规定？"))
        elif intent in HIGH_SIGNAL_INTENTS:
            queries.append(("chunk", f"{scope}里关于{intent}是怎么说明的？"))
    if not queries and _is_valid_section_hint(section_hint) and len(section_hint) >= 12:
        queries.append(("summary", f"{scope}中关于{section_hint}主要介绍了什么？"))
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for target_mode, query in queries:
        key = (target_mode, query)
        if query and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _build_constraint_queries(*, scope: str, topic: str, intents: list[str], constraints: list[str]) -> list[str]:
    queries: list[str] = []
    year = _select_primary_year(scope=scope, topic=topic, constraints=constraints)
    year_prefix = f"{year}年" if year and year not in scope else ""
    if "学费" in intents:
        queries.append(f"{year_prefix}{scope}的学费标准是多少？")
    if "住宿费" in intents:
        queries.append(f"{year_prefix}{scope}的住宿费标准是多少？")
    if "电话" in intents:
        queries.append(f"{scope}的咨询电话是多少？")
    if "网址" in intents:
        queries.append(f"{scope}的官方网站是什么？")
    if "地址" in intents:
        queries.append(f"{scope}的地址在哪里？")
    if "选科" in intents:
        queries.append(f"{scope}的选考科目要求是什么？")
    if "录取" in intents and year:
        queries.append(f"{year_prefix}{scope}的录取要求是什么？")
    if ("奖学金" in intents or "资助" in intents) and topic:
        queries.append(f"{topic}有哪些资助或奖学金政策？")
    return list(dict.fromkeys(query for query in queries if query))


def _build_lexical_queries(*, scope: str, topic: str, section_hint: str, intents: list[str], constraints: list[str], expansions: list[str]) -> list[str]:
    queries: list[str] = []
    compact_scope = _sanitize_label(scope, limit=18)
    compact_topic = _sanitize_label(topic, limit=14)
    compact_section = _sanitize_label(section_hint, limit=14)
    for intent in intents[:2]:
        if intent not in HIGH_SIGNAL_INTENTS:
            continue
        if intent == "电话":
            queries.append(f"{compact_scope} 咨询电话")
        elif intent == "网址":
            queries.append(f"{compact_scope} 官方网站")
        elif intent == "地址":
            queries.append(f"{compact_scope} 地址")
        elif intent == "学费":
            queries.append(f"{compact_scope} 学费标准")
        elif intent == "住宿费":
            queries.append(f"{compact_scope} 住宿费标准")
        elif intent == "录取":
            queries.append(f"{compact_scope} 录取要求")
        elif intent == "选科":
            queries.append(f"{compact_scope} 选考科目")
        elif intent == "资助" or intent == "奖学金":
            queries.append(f"{compact_scope} 资助政策")
    for expansion in expansions:
        cleaned = _sanitize_label(str(expansion), limit=20)
        if cleaned and re.search(r"(学费|住宿|电话|地址|官网|网址|录取|选考|资助|奖学金|贷款)", cleaned):
            queries.append(cleaned)
    return list(dict.fromkeys(query for query in queries if len(query.strip()) >= 4))


def _build_relevant_chunk_ids(*, style: str, target_mode: str, chunk_id: str, summary_chunk_id: str, sibling_chunk_ids: list[str]) -> list[str]:
    if style == "semantic":
        ordered = [summary_chunk_id] if target_mode == "summary" and summary_chunk_id else [chunk_id]
    elif style == "constraint":
        ordered = [chunk_id]
    else:
        ordered = [chunk_id]
    return [item for item in dict.fromkeys(ordered) if item]


def _estimate_case_quality(*, style: str, query: str, chunk_text: str, section_hint: str, source_title: str, target_mode: str) -> float:
    text = " ".join([chunk_text, section_hint, source_title])
    score = 0.0
    if style == "semantic" and target_mode == "summary":
        if _is_valid_section_hint(section_hint):
            score += 1.2
        if section_hint and section_hint in query:
            score += 0.8
    intent_rules = [
        ("电话", ("电话", "咨询电话", "联系方式", "0371")),
        ("地址", ("地址", "校区", "路", "号")),
        ("学费", ("学费", "收费", "元/生", "元/年")),
        ("住宿费", ("住宿费", "住宿标准", "元/生")),
        ("网址", ("网址", "官网", "https://", "http://")),
        ("录取", ("录取", "投档", "调档", "艺术类", "专业分流")),
        ("资助", ("资助", "奖学金", "贷款")),
        ("奖学金", ("奖学金", "资助", "贷款")),
        ("选考", ("选考", "科目要求")),
    ]
    for intent, aliases in intent_rules:
        if intent not in query and not any(alias in query for alias in aliases):
            continue
        score += sum(0.45 for alias in aliases if alias in text)
    score += min(0.6, len(re.findall(r"\d+", chunk_text)) * 0.1)
    if style == "lexical":
        score += 0.2
    return round(score, 4)


def _build_case_candidates(
    *,
    doc: Document,
    topic: str,
    sibling_chunk_ids: list[str],
    summary_chunk_id: str,
    candidate_map: dict[tuple[str, str, str], tuple[float, AblationCase]],
) -> None:
    chunk_id = str(doc.metadata.get("chunk_id", "")).strip()
    parent_id = str(doc.metadata.get("parent_id", "")).strip() or chunk_id
    chunk_text = str(doc.metadata.get("chunk_text", "")).strip()
    source_title = _sanitize_label(str(doc.metadata.get("source_title", "")).strip())
    section_hint = _sanitize_label(str(doc.metadata.get("section_hint", "")).strip())
    scope = source_title or topic or section_hint or "招生资料"
    intents = _detect_query_intents(chunk_text=chunk_text, topic=topic, source_title=source_title, section_hint=section_hint)
    constraints = _extract_query_constraints(chunk_text)
    candidates: list[tuple[str, str, str]] = []
    candidates.extend(("semantic", query, target_mode) for target_mode, query in _build_semantic_queries(scope=scope, topic=topic, section_hint=section_hint, intents=intents))
    candidates.extend(("constraint", query, "chunk") for query in _build_constraint_queries(scope=scope, topic=topic, intents=intents, constraints=constraints))
    expansions = doc.metadata.get("query_expansions", [])
    normalized_expansions = [str(query).strip() for query in expansions] if isinstance(expansions, list) else []
    candidates.extend(
        ("lexical", query, "chunk")
        for query in _build_lexical_queries(
            scope=scope,
            topic=topic,
            section_hint=section_hint,
            intents=intents,
            constraints=constraints,
            expansions=normalized_expansions,
        )
    )

    style_counts: dict[str, int] = {"semantic": 0, "constraint": 0, "lexical": 0}
    for style, query, target_mode in candidates:
        normalized = _normalize_query(query)
        if len(normalized) < 4:
            continue
        if style_counts.get(style, 0) >= 1:
            continue
        dedupe_key = (parent_id, style, normalized)
        style_counts[style] += 1
        relevant_chunk_ids = _build_relevant_chunk_ids(
            style=style,
            target_mode=target_mode,
            chunk_id=chunk_id,
            summary_chunk_id=summary_chunk_id,
            sibling_chunk_ids=sibling_chunk_ids,
        )
        case = AblationCase(
            name=f"{chunk_id}-{style}",
            category=topic,
            query=normalized,
            relevant_chunk_ids=relevant_chunk_ids,
            query_style=style,
        )
        quality = _estimate_case_quality(
            style=style,
            query=normalized,
            chunk_text=chunk_text,
            section_hint=section_hint,
            source_title=source_title,
            target_mode=target_mode,
        )
        best = candidate_map.get(dedupe_key)
        if best is None or quality > best[0]:
            candidate_map[dedupe_key] = (quality, case)


def _round_robin_cases(buckets: dict[str, list[AblationCase]], max_cases: int) -> list[AblationCase]:
    ordered: list[AblationCase] = []
    while len(ordered) < max_cases:
        progressed = False
        for style in STYLE_ORDER:
            queue = buckets.get(style, [])
            if not queue:
                continue
            ordered.append(queue.pop(0))
            progressed = True
            if len(ordered) >= max_cases:
                break
        if not progressed:
            break
    return ordered


def build_balanced_cases(docs: list[Document], max_cases: int = 48) -> list[AblationCase]:
    by_parent: dict[str, list[str]] = {}
    summary_by_parent: dict[str, str] = {}
    candidate_map: dict[tuple[str, str, str], tuple[float, AblationCase]] = {}

    for doc in docs:
        parent_id = str(doc.metadata.get("parent_id", "")).strip()
        chunk_id = str(doc.metadata.get("chunk_id", "")).strip()
        chunk_level = str(doc.metadata.get("chunk_level", "")).strip()
        if not parent_id or not chunk_id:
            continue
        if chunk_level == "summary":
            summary_by_parent[parent_id] = chunk_id
            continue
        by_parent.setdefault(parent_id, []).append(chunk_id)

    for doc in docs:
        if str(doc.metadata.get("chunk_level", "")) != "small":
            continue
        chunk_id = str(doc.metadata.get("chunk_id", "")).strip()
        parent_id = str(doc.metadata.get("parent_id", "")).strip()
        topic = str(doc.metadata.get("topic", "")).strip() or "未分类"
        if not chunk_id or not parent_id:
            continue
        sibling_chunk_ids = [item for item in by_parent.get(parent_id, []) if item and item != chunk_id]
        summary_chunk_id = summary_by_parent.get(parent_id, "")
        _build_case_candidates(
            doc=doc,
            topic=topic,
            sibling_chunk_ids=sibling_chunk_ids,
            summary_chunk_id=summary_chunk_id,
            candidate_map=candidate_map,
        )
    buckets: dict[str, list[AblationCase]] = {style: [] for style in STYLE_ORDER}
    for _, case in sorted(candidate_map.values(), key=lambda item: (item[0], item[1].name), reverse=True):
        buckets[case.query_style].append(case)
    return _round_robin_cases(buckets=buckets, max_cases=max_cases)


def _clone_doc_with_score(doc: Document, score: float | None = None) -> Document:
    cloned = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
    if score is not None:
        cloned.metadata["score"] = float(score)
    return cloned


def _classify_query_shape(query: str) -> str:
    normalized = query.strip()
    if any(marker in normalized for marker in SEMANTIC_QUERY_MARKERS):
        return "semantic"
    if any(marker in normalized for marker in CONSTRAINT_QUERY_MARKERS):
        return "constraint"
    if any(char.isdigit() for char in normalized):
        return "constraint"
    return "lexical"


def _variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec(name="feature_off", kind="hybrid", sparse_weight=1.0, dense_weight=1.0, exact_weight=1.2, rrf_k=60),
        VariantSpec(name="feature_on", kind="adaptive"),
        VariantSpec(name="exact_only", kind="exact"),
        VariantSpec(name="dense_only", kind="dense"),
        VariantSpec(name="bm25_only", kind="bm25"),
        VariantSpec(name="hybrid_tuned", kind="hybrid", sparse_weight=0.6, dense_weight=0.8, exact_weight=2.4, rrf_k=20),
    ]


def _ensure_retriever(
    cache: dict[tuple[float, float, float, int], HybridRetriever],
    index: RagIndexManager,
    spec: VariantSpec,
) -> HybridRetriever:
    key = (spec.sparse_weight, spec.dense_weight, spec.exact_weight, spec.rrf_k)
    if key not in cache:
        cache[key] = HybridRetriever(
            index=index,
            sparse_weight=spec.sparse_weight,
            dense_weight=spec.dense_weight,
            exact_weight=spec.exact_weight,
            rrf_k=spec.rrf_k,
        )
    return cache[key]


def _predict_documents(
    *,
    index: RagIndexManager,
    retriever_cache: dict[tuple[float, float, float, int], HybridRetriever],
    spec: VariantSpec,
    query: str,
    top_k: int,
) -> tuple[list[Document], str]:
    if spec.kind == "adaptive":
        normalized = query.strip()
        if (
            _classify_query_shape(query) in {"lexical", "constraint"}
            or any(intent in normalized for intent in HIGH_SIGNAL_INTENTS)
            or "联系方式" in normalized
        ):
            selected = "exact_only"
        else:
            selected = "hybrid_tuned"
        delegated = next(item for item in _variant_specs() if item.name == selected)
        docs, _ = _predict_documents(index=index, retriever_cache=retriever_cache, spec=delegated, query=query, top_k=top_k)
        return docs, selected
    if spec.kind == "hybrid":
        retriever = _ensure_retriever(retriever_cache, index, spec)
        rows = retriever.retrieve(queries=[query], top_n=top_k)
        return [
            _clone_doc_with_score(item.document, score=item.score)
            for item in rows
            if str(item.document.metadata.get("chunk_id", ""))
        ], spec.name
    if spec.kind == "dense":
        rows = index.dense_similarity_scores(query=query, top_k=top_k)
        return [_clone_doc_with_score(doc, score=score) for doc, score in rows if str(doc.metadata.get("chunk_id", ""))], spec.name
    if spec.kind == "bm25":
        rows = index.get_bm25_retriever(top_k=top_k).invoke(query)
        return [_clone_doc_with_score(doc, score=doc.metadata.get("score")) for doc in rows if str(doc.metadata.get("chunk_id", ""))], spec.name
    if spec.kind == "exact":
        base = _ensure_retriever(retriever_cache, index, VariantSpec(name="exact_base", kind="hybrid"))
        rows = base._exact_match_search(query=query, top_n=top_k)  # noqa: SLF001
        return [_clone_doc_with_score(doc, score=doc.metadata.get("score")) for doc in rows if str(doc.metadata.get("chunk_id", ""))], spec.name
    raise ValueError(f"unknown variant kind: {spec.kind}")


def _summarize_rows(rows: list[dict], k: int) -> dict[str, object]:
    if not rows:
        return {
            "cases": 0,
            f"recall@{k}": 0.0,
            f"mrr@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            "avg_latency_ms": 0.0,
            "query_style_breakdown": {},
            "selected_backend_breakdown": {},
        }
    style_breakdown: dict[str, dict[str, float | int]] = {}
    backend_breakdown: dict[str, int] = {}
    for row in rows:
        style = str(row["query_style"])
        backend = str(row["selected_backend"])
        backend_breakdown[backend] = backend_breakdown.get(backend, 0) + 1
        bucket = style_breakdown.setdefault(
            style,
            {"cases": 0, f"recall@{k}": 0.0, f"mrr@{k}": 0.0, f"ndcg@{k}": 0.0},
        )
        bucket["cases"] += 1
        bucket[f"recall@{k}"] += float(row[f"recall@{k}"])
        bucket[f"mrr@{k}"] += float(row[f"mrr@{k}"])
        bucket[f"ndcg@{k}"] += float(row[f"ndcg@{k}"])
    for bucket in style_breakdown.values():
        total = int(bucket["cases"])
        bucket[f"recall@{k}"] = round(float(bucket[f"recall@{k}"]) / total, 4)
        bucket[f"mrr@{k}"] = round(float(bucket[f"mrr@{k}"]) / total, 4)
        bucket[f"ndcg@{k}"] = round(float(bucket[f"ndcg@{k}"]) / total, 4)
    return {
        "cases": len(rows),
        f"recall@{k}": round(sum(float(row[f"recall@{k}"]) for row in rows) / len(rows), 4),
        f"mrr@{k}": round(sum(float(row[f"mrr@{k}"]) for row in rows) / len(rows), 4),
        f"ndcg@{k}": round(sum(float(row[f"ndcg@{k}"]) for row in rows) / len(rows), 4),
        "avg_latency_ms": round(sum(float(row["latency_ms"]) for row in rows) / len(rows), 4),
        "query_style_breakdown": style_breakdown,
        "selected_backend_breakdown": backend_breakdown,
    }


def _evaluate_variant(
    *,
    index: RagIndexManager,
    retriever_cache: dict[tuple[float, float, float, int], HybridRetriever],
    spec: VariantSpec,
    cases: list[AblationCase],
    k: int,
) -> dict[str, object]:
    rows: list[dict] = []
    for case in cases:
        start = time.perf_counter()
        docs, selected_backend = _predict_documents(
            index=index,
            retriever_cache=retriever_cache,
            spec=spec,
            query=case.query,
            top_k=k,
        )
        predicted_ids = [str(doc.metadata.get("chunk_id", "")) for doc in docs if str(doc.metadata.get("chunk_id", ""))]
        relevant_ids = set(case.relevant_chunk_ids)
        rows.append(
            {
                "name": case.name,
                "category": case.category,
                "query": case.query,
                "query_style": case.query_style,
                "selected_backend": selected_backend,
                f"recall@{k}": compute_recall_at_k(relevant_ids, predicted_ids, k),
                f"mrr@{k}": compute_mrr_at_k(relevant_ids, predicted_ids, k),
                f"ndcg@{k}": compute_ndcg_at_k(relevant_ids, predicted_ids, k),
                "latency_ms": round((time.perf_counter() - start) * 1000, 4),
            }
        )
    summary = _summarize_rows(rows=rows, k=k)
    summary["variant"] = spec.name
    return summary


def _attach_research_score(variants: list[dict[str, object]], k: int) -> None:
    for item in variants:
        recall_score = float(item[f"recall@{k}"])
        mrr_score = float(item[f"mrr@{k}"])
        ndcg_score = float(item[f"ndcg@{k}"])
        latency_ms = max(float(item["avg_latency_ms"]), 0.0)
        latency_score = 1.0 / (1.0 + latency_ms / 1000.0)
        research_score = (
            0.25 * recall_score
            + 0.43 * mrr_score
            + 0.3 * ndcg_score
            + 0.02 * latency_score
        )
        item["research_score"] = round(research_score, 4)
        item["research_score_components"] = {
            f"recall@{k}": round(recall_score, 4),
            f"mrr@{k}": round(mrr_score, 4),
            f"ndcg@{k}": round(ndcg_score, 4),
            "latency_score": round(latency_score, 4),
        }


def main() -> int:
    settings = get_settings()
    index = RagIndexManager(settings)
    index.startup()
    docs = index.all_documents()
    cases = build_balanced_cases(docs=docs, max_cases=18)
    k = 5
    retriever_cache: dict[tuple[float, float, float, int], HybridRetriever] = {}
    variants = [
        _evaluate_variant(index=index, retriever_cache=retriever_cache, spec=spec, cases=cases, k=k)
        for spec in _variant_specs()
    ]
    _attach_research_score(variants=variants, k=k)
    variants.sort(key=lambda item: (float(item["research_score"]), float(item[f"mrr@{k}"]), float(item[f"ndcg@{k}"])), reverse=True)
    off = next(item for item in variants if str(item["variant"]) == "feature_off")
    on = next(item for item in variants if str(item["variant"]) == "feature_on")
    report = {
        "status": "ok" if cases else "skipped",
        "cases": len(cases),
        "k": k,
        "case_manifest": [asdict(case) for case in cases],
        "variants": variants,
        "feature_delta": {
            "feature_off_score": off["research_score"],
            "feature_on_score": on["research_score"],
            "score_gain": round(float(on["research_score"]) - float(off["research_score"]), 4),
            f"mrr_gain@{k}": round(float(on[f"mrr@{k}"]) - float(off[f"mrr@{k}"]), 4),
            f"ndcg_gain@{k}": round(float(on[f"ndcg@{k}"]) - float(off[f"ndcg@{k}"]), 4),
        },
        "best_variant": variants[0]["variant"] if variants else "",
    }
    output = ROOT / "reports" / "retrieval_ablation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

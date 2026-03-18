from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag.index import RagIndexManager
from app.rag.retrievers import HybridRetriever


@dataclass(slots=True)
class RetrievalEvalCase:
    name: str
    category: str
    query: str
    relevant_chunk_ids: list[str]


def compute_recall_at_k(relevant_ids: set[str], predicted_ids: list[str], k: int) -> float:
    if not relevant_ids or k <= 0:
        return 0.0
    hits = sum(1 for chunk_id in predicted_ids[:k] if chunk_id in relevant_ids)
    return round(hits / len(relevant_ids), 4)


def compute_mrr_at_k(relevant_ids: set[str], predicted_ids: list[str], k: int) -> float:
    if not relevant_ids or k <= 0:
        return 0.0
    for rank, chunk_id in enumerate(predicted_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return round(1.0 / rank, 4)
    return 0.0


def compute_ndcg_at_k(relevant_ids: set[str], predicted_ids: list[str], k: int) -> float:
    if not relevant_ids or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, chunk_id in enumerate(predicted_ids[:k], start=1):
        if chunk_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return round(dcg / idcg, 4)


def build_retrieval_cases(docs: list, max_cases: int = 48) -> list[RetrievalEvalCase]:
    cases: list[RetrievalEvalCase] = []
    seen: set[tuple[str, str]] = set()
    by_parent: dict[str, list[str]] = {}
    summary_by_parent: dict[str, str] = {}
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
        relevant = list(dict.fromkeys([chunk_id, *by_parent.get(parent_id, []), summary_by_parent.get(parent_id, "")]))
        relevant = [item for item in relevant if item]
        expansions = doc.metadata.get("query_expansions", [])
        candidates: list[str] = []
        if isinstance(expansions, list):
            candidates.extend(str(item).strip() for item in expansions if str(item).strip())
        chunk_text = str(doc.metadata.get("chunk_text", "")).strip()
        if chunk_text:
            candidates.append(chunk_text[:32])
        for idx, query in enumerate(candidates, start=1):
            normalized = " ".join(query.split())
            if len(normalized) < 4:
                continue
            dedupe_key = (chunk_id, normalized)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            cases.append(
                RetrievalEvalCase(
                    name=f"{chunk_id}-{idx}",
                    category=topic,
                    query=normalized,
                    relevant_chunk_ids=relevant,
                )
            )
            if len(cases) >= max_cases:
                return cases
    return cases


def build_p95_latency(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    ordered = sorted(float(row["latency_ms"]) for row in rows)
    index = max(math.ceil(len(ordered) * 0.95) - 1, 0)
    return round(ordered[index], 4)


def summarize_metrics(rows: list[dict], k: int) -> dict[str, float]:
    if not rows:
        return {
            f"recall@{k}": 0.0,
            f"mrr@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    return {
        f"recall@{k}": round(sum(float(row[f"recall@{k}"]) for row in rows) / len(rows), 4),
        f"mrr@{k}": round(sum(float(row[f"mrr@{k}"]) for row in rows) / len(rows), 4),
        f"ndcg@{k}": round(sum(float(row[f"ndcg@{k}"]) for row in rows) / len(rows), 4),
        "avg_latency_ms": round(sum(float(row["latency_ms"]) for row in rows) / len(rows), 4),
        "p95_latency_ms": build_p95_latency(rows),
    }


def run_case(retriever: HybridRetriever, case: RetrievalEvalCase, k: int) -> dict:
    start = time.perf_counter()
    rows = retriever.retrieve(queries=[case.query], top_n=k)
    elapsed = (time.perf_counter() - start) * 1000
    predicted_ids = [str(item.document.metadata.get("chunk_id", "")) for item in rows if str(item.document.metadata.get("chunk_id", ""))]
    relevant_ids = set(case.relevant_chunk_ids)
    return {
        "name": case.name,
        "category": case.category,
        "query": case.query,
        f"recall@{k}": compute_recall_at_k(relevant_ids, predicted_ids, k),
        f"mrr@{k}": compute_mrr_at_k(relevant_ids, predicted_ids, k),
        f"ndcg@{k}": compute_ndcg_at_k(relevant_ids, predicted_ids, k),
        "latency_ms": round(elapsed, 4),
        "predicted_chunk_ids": predicted_ids[:k],
        "relevant_chunk_ids": case.relevant_chunk_ids,
    }


def build_bucket_summary(rows: list[dict], k: int) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, dict[str, float | int]] = {}
    for row in rows:
        bucket = buckets.setdefault(
            str(row["category"]),
            {
                "total": 0,
                f"recall@{k}": 0.0,
                f"mrr@{k}": 0.0,
                f"ndcg@{k}": 0.0,
            },
        )
        bucket["total"] += 1
        bucket[f"recall@{k}"] += float(row[f"recall@{k}"])
        bucket[f"mrr@{k}"] += float(row[f"mrr@{k}"])
        bucket[f"ndcg@{k}"] += float(row[f"ndcg@{k}"])
    for bucket in buckets.values():
        total = int(bucket["total"])
        if total <= 0:
            continue
        bucket[f"recall@{k}"] = round(float(bucket[f"recall@{k}"]) / total, 4)
        bucket[f"mrr@{k}"] = round(float(bucket[f"mrr@{k}"]) / total, 4)
        bucket[f"ndcg@{k}"] = round(float(bucket[f"ndcg@{k}"]) / total, 4)
    return buckets


def main() -> int:
    settings = get_settings()
    index = RagIndexManager(settings)
    index.startup()
    docs = index.all_documents()
    retriever = HybridRetriever(index=index)
    k = 5
    cases = build_retrieval_cases(docs=docs, max_cases=48)
    rows = [run_case(retriever=retriever, case=case, k=k) for case in cases]
    summary = {
        "status": "ok" if rows else "skipped",
        "cases": len(cases),
        "k": k,
        "metrics": summarize_metrics(rows=rows, k=k),
        "bucket_summary": build_bucket_summary(rows=rows, k=k),
        "rows": rows,
        "case_manifest": [asdict(case) for case in cases],
    }
    output = Path("reports") / "retrieval_eval_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

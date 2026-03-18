from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag.index import RagIndexManager
from app.rag.retrievers import HybridRetriever
from scripts.evaluate_retrieval import (
    RetrievalEvalCase,
    build_retrieval_cases,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
)


def _predict_chunk_ids(retriever: HybridRetriever, variant: str, query: str, k: int) -> list[str]:
    if variant == "rrf_hybrid":
        rows = retriever.retrieve(queries=[query], top_n=k)
        return [str(item.document.metadata.get("chunk_id", "")) for item in rows if str(item.document.metadata.get("chunk_id", ""))]
    if variant == "bm25_only":
        rows = retriever.index.get_bm25_retriever(top_k=k).invoke(query)
        return [str(doc.metadata.get("chunk_id", "")) for doc in rows if str(doc.metadata.get("chunk_id", ""))]
    if variant == "dense_only":
        rows = retriever.index.dense_similarity_scores(query=query, top_k=k)
        return [str(doc.metadata.get("chunk_id", "")) for doc, _ in rows if str(doc.metadata.get("chunk_id", ""))]
    if variant == "exact_only":
        rows = retriever._exact_match_search(query=query, top_n=k)  # noqa: SLF001
        return [str(doc.metadata.get("chunk_id", "")) for doc in rows if str(doc.metadata.get("chunk_id", ""))]
    raise ValueError(f"unknown variant: {variant}")


def evaluate_variant(
    retriever: HybridRetriever,
    cases: list[RetrievalEvalCase],
    variant: str,
    k: int,
) -> dict:
    rows: list[dict] = []
    for case in cases:
        relevant_ids = set(case.relevant_chunk_ids)
        start = time.perf_counter()
        predicted_ids = _predict_chunk_ids(retriever=retriever, variant=variant, query=case.query, k=k)
        latency_ms = round((time.perf_counter() - start) * 1000, 4)
        rows.append(
            {
                "name": case.name,
                "category": case.category,
                "query": case.query,
                f"recall@{k}": compute_recall_at_k(relevant_ids, predicted_ids, k),
                f"mrr@{k}": compute_mrr_at_k(relevant_ids, predicted_ids, k),
                f"ndcg@{k}": compute_ndcg_at_k(relevant_ids, predicted_ids, k),
                "latency_ms": latency_ms,
            }
        )
    return summarize_variant(variant=variant, rows=rows, k=k)


def summarize_variant(variant: str, rows: list[dict], k: int) -> dict[str, float | int | str]:
    if not rows:
        return {
            "variant": variant,
            "cases": 0,
            f"recall@{k}": 0.0,
            f"mrr@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            "avg_latency_ms": 0.0,
        }
    return {
        "variant": variant,
        "cases": len(rows),
        f"recall@{k}": round(sum(float(row[f"recall@{k}"]) for row in rows) / len(rows), 4),
        f"mrr@{k}": round(sum(float(row[f"mrr@{k}"]) for row in rows) / len(rows), 4),
        f"ndcg@{k}": round(sum(float(row[f"ndcg@{k}"]) for row in rows) / len(rows), 4),
        "avg_latency_ms": round(sum(float(row["latency_ms"]) for row in rows) / len(rows), 4),
    }


def main() -> int:
    settings = get_settings()
    index = RagIndexManager(settings)
    index.startup()
    retriever = HybridRetriever(index=index)
    cases = build_retrieval_cases(index.all_documents(), max_cases=48)
    k = 5
    variants = ["rrf_hybrid", "bm25_only", "dense_only", "exact_only"]
    summary_rows = [evaluate_variant(retriever=retriever, cases=cases, variant=variant, k=k) for variant in variants]
    report = {
        "status": "ok" if cases else "skipped",
        "cases": len(cases),
        "k": k,
        "variants": summary_rows,
    }
    output = Path("reports") / "retrieval_ablation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

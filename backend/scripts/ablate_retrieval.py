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
from app.rag.rewrite import QueryRewriter
from scripts.evaluate_retrieval import (
    RetrievalEvalCase,
    build_retrieval_cases,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
)


def build_variant_queries(case: RetrievalEvalCase, variant: str, rewriter: QueryRewriter) -> list[str]:
    if not variant.endswith("_rewrite_on"):
        return [case.query]
    rewritten = rewriter.rewrite(case.query)
    if case.query not in rewritten:
        rewritten.insert(0, case.query)
    return list(dict.fromkeys(item.strip() for item in rewritten if item.strip()))[:3]


def _predict_chunk_ids(retriever: HybridRetriever, variant: str, queries: list[str], k: int) -> list[str]:
    base_variant = variant.replace("_rewrite_on", "").replace("_rewrite_off", "")
    query = queries[0] if queries else ""
    if base_variant == "rrf_hybrid":
        rows = retriever.retrieve(queries=queries or [query], top_n=k)
        return [str(item.document.metadata.get("chunk_id", "")) for item in rows if str(item.document.metadata.get("chunk_id", ""))]
    if base_variant == "bm25_only":
        rows = retriever.index.get_bm25_retriever(top_k=k).invoke(query)
        return [str(doc.metadata.get("chunk_id", "")) for doc in rows if str(doc.metadata.get("chunk_id", ""))]
    if base_variant == "dense_only":
        rows = retriever.index.dense_similarity_scores(query=query, top_k=k)
        return [str(doc.metadata.get("chunk_id", "")) for doc, _ in rows if str(doc.metadata.get("chunk_id", ""))]
    if base_variant == "exact_only":
        rows = retriever._exact_match_search(query=query, top_n=k)  # noqa: SLF001
        return [str(doc.metadata.get("chunk_id", "")) for doc in rows if str(doc.metadata.get("chunk_id", ""))]
    raise ValueError(f"unknown variant: {variant}")


def evaluate_variant(
    retriever: HybridRetriever,
    rewriter: QueryRewriter,
    cases: list[RetrievalEvalCase],
    variant: str,
    k: int,
) -> dict:
    rows: list[dict] = []
    for case in cases:
        relevant_ids = set(case.relevant_chunk_ids)
        queries = build_variant_queries(case=case, variant=variant, rewriter=rewriter)
        start = time.perf_counter()
        predicted_ids = _predict_chunk_ids(retriever=retriever, variant=variant, queries=queries, k=k)
        latency_ms = round((time.perf_counter() - start) * 1000, 4)
        rows.append(
            {
                "name": case.name,
                "category": case.category,
                "query": case.query,
                "queries": queries,
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
    rewriter = QueryRewriter(settings)
    cases = build_retrieval_cases(index.all_documents(), max_cases=48)
    k = 5
    variants = [
        "rrf_hybrid_rewrite_off",
        "rrf_hybrid_rewrite_on",
        "bm25_only",
        "dense_only",
        "exact_only",
    ]
    summary_rows = [
        evaluate_variant(retriever=retriever, rewriter=rewriter, cases=cases, variant=variant, k=k)
        for variant in variants
    ]
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

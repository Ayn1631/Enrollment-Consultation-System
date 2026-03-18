from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag.index import RagIndexManager
from app.rag.rerank import ListwiseReranker
from app.rag.retrievers import HybridRetriever
from app.rag.rewrite import QueryRewriter
from scripts.evaluate_retrieval import (
    RetrievalEvalCase,
    build_retrieval_cases,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
)


def _variant_flag_enabled(variant: str, name: str) -> bool:
    return f"_{name}_on" in variant


def _base_variant_name(variant: str) -> str:
    base = variant
    for suffix in ("_rewrite_on", "_rewrite_off", "_rerank_on", "_rerank_off"):
        base = base.replace(suffix, "")
    return base


def build_variant_queries(case: RetrievalEvalCase, variant: str, rewriter: QueryRewriter) -> list[str]:
    if not _variant_flag_enabled(variant, "rewrite"):
        return [case.query]
    rewritten = rewriter.rewrite(case.query)
    if case.query not in rewritten:
        rewritten.insert(0, case.query)
    return list(dict.fromkeys(item.strip() for item in rewritten if item.strip()))[:3]


def _clone_doc_with_score(doc: Document, score: float | None = None) -> Document:
    cloned = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
    if score is not None:
        try:
            cloned.metadata["score"] = float(score)
        except (TypeError, ValueError):
            pass
    return cloned


def _predict_documents(retriever: HybridRetriever, variant: str, queries: list[str], k: int) -> list[Document]:
    base_variant = _base_variant_name(variant)
    query = queries[0] if queries else ""
    if base_variant == "rrf_hybrid":
        rows = retriever.retrieve(queries=queries or [query], top_n=k)
        return [
            _clone_doc_with_score(item.document, score=item.score)
            for item in rows
            if str(item.document.metadata.get("chunk_id", ""))
        ]
    if base_variant == "bm25_only":
        rows = retriever.index.get_bm25_retriever(top_k=k).invoke(query)
        return [_clone_doc_with_score(doc, score=doc.metadata.get("score")) for doc in rows if str(doc.metadata.get("chunk_id", ""))]
    if base_variant == "dense_only":
        rows = retriever.index.dense_similarity_scores(query=query, top_k=k)
        return [_clone_doc_with_score(doc, score=score) for doc, score in rows if str(doc.metadata.get("chunk_id", ""))]
    if base_variant == "exact_only":
        rows = retriever._exact_match_search(query=query, top_n=k)  # noqa: SLF001
        return [_clone_doc_with_score(doc, score=doc.metadata.get("score")) for doc in rows if str(doc.metadata.get("chunk_id", ""))]
    raise ValueError(f"unknown variant: {variant}")


def _predict_chunk_ids(docs: list[Document]) -> list[str]:
    return [str(doc.metadata.get("chunk_id", "")) for doc in docs if str(doc.metadata.get("chunk_id", ""))]


def evaluate_variant(
    retriever: HybridRetriever,
    rewriter: QueryRewriter,
    reranker: ListwiseReranker,
    cases: list[RetrievalEvalCase],
    variant: str,
    k: int,
) -> dict:
    rows: list[dict] = []
    for case in cases:
        relevant_ids = set(case.relevant_chunk_ids)
        queries = build_variant_queries(case=case, variant=variant, rewriter=rewriter)
        start = time.perf_counter()
        top_n = max(k * 3, 12) if _variant_flag_enabled(variant, "rerank") else k
        docs = _predict_documents(retriever=retriever, variant=variant, queries=queries, k=top_n)
        if _variant_flag_enabled(variant, "rerank"):
            docs, _ = reranker.rerank(query=queries[0] if queries else case.query, docs=docs, top_k=k)
        predicted_ids = _predict_chunk_ids(docs)
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
    reranker = ListwiseReranker(settings)
    cases = build_retrieval_cases(index.all_documents(), max_cases=48)
    k = 5
    variants = [
        "rrf_hybrid_rewrite_off",
        "rrf_hybrid_rewrite_on",
        "rrf_hybrid_rerank_off",
        "rrf_hybrid_rerank_on",
        "bm25_only",
        "dense_only",
        "exact_only",
    ]
    summary_rows = [
        evaluate_variant(retriever=retriever, rewriter=rewriter, reranker=reranker, cases=cases, variant=variant, k=k)
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

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag.index import RagIndexManager


@dataclass(slots=True)
class AnnTuningCase:
    query: str
    target_chunk_id: str


@dataclass(slots=True)
class AnnTuningRow:
    m: int
    ef_construction: int
    ef_search: int
    avg_recall_at_k: float
    p95_latency_ms: float
    avg_latency_ms: float


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            continue
    return values or default


def build_tuning_cases(docs: list, max_cases: int) -> list[AnnTuningCase]:
    cases: list[AnnTuningCase] = []
    seen_queries: set[str] = set()
    for doc in docs:
        if str(doc.metadata.get("chunk_level", "")) != "small":
            continue
        chunk_id = str(doc.metadata.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        candidates: list[str] = []
        expansions = doc.metadata.get("query_expansions", [])
        if isinstance(expansions, list):
            candidates.extend(str(item).strip() for item in expansions if str(item).strip())
        chunk_text = str(doc.metadata.get("chunk_text", "")).strip()
        if chunk_text:
            candidates.append(chunk_text[:32])
        for candidate in candidates:
            normalized = " ".join(candidate.split())
            if len(normalized) < 4 or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            cases.append(AnnTuningCase(query=normalized, target_chunk_id=chunk_id))
            if len(cases) >= max_cases:
                return cases
    return cases


def compute_recall_at_k(relevant_ids: set[str], predicted_ids: list[str]) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for chunk_id in predicted_ids if chunk_id in relevant_ids)
    return round(hits / len(relevant_ids), 4)


def summarize_curve(rows: list[AnnTuningRow]) -> list[dict[str, float | int]]:
    frontier: list[dict[str, float | int]] = []
    best_recall = -1.0
    for row in sorted(rows, key=lambda item: (item.avg_latency_ms, -item.avg_recall_at_k)):
        if row.avg_recall_at_k <= best_recall:
            continue
        best_recall = row.avg_recall_at_k
        frontier.append(
            {
                "m": row.m,
                "ef_construction": row.ef_construction,
                "ef_search": row.ef_search,
                "avg_recall_at_k": row.avg_recall_at_k,
                "avg_latency_ms": row.avg_latency_ms,
                "p95_latency_ms": row.p95_latency_ms,
            }
        )
    return frontier


def _top_k_by_dot_product(doc_vectors, query_vector, top_k: int) -> list[int]:
    scored = [(idx, float(sum(left * right for left, right in zip(query_vector, doc_vector)))) for idx, doc_vector in enumerate(doc_vectors)]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in scored[:top_k]]


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(int(len(ordered) * ratio + 0.9999) - 1, 0)
    return round(ordered[index], 4)


def main() -> int:
    try:
        import faiss  # type: ignore
        import numpy as np
    except Exception:
        report = {
            "status": "skipped",
            "reason": "faiss_or_numpy_unavailable",
        }
        output = Path(os.getenv("ANN_TUNE_REPORT_PATH", "reports/ann_tuning_report.json"))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    settings = get_settings()
    index = RagIndexManager(settings)
    index.startup()
    docs = index.all_documents()
    max_cases = _read_int("ANN_TUNE_MAX_CASES", 24)
    top_k = _read_int("ANN_TUNE_TOP_K", 5)
    cases = build_tuning_cases(docs=docs, max_cases=max_cases)
    if not cases:
        report = {
            "status": "skipped",
            "reason": "no_tuning_cases",
        }
        output = Path(os.getenv("ANN_TUNE_REPORT_PATH", "reports/ann_tuning_report.json"))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if getattr(index, "_local_dense_vectors", None):
        doc_vectors = [list(vector) for vector in index._local_dense_vectors]
    else:
        doc_vectors = index._embeddings.embed_documents([doc.page_content for doc in docs])
    query_vectors = [index._embeddings.embed_query(case.query) for case in cases]
    chunk_ids = [str(doc.metadata.get("chunk_id", "")) for doc in docs]
    ground_truth = [
        {chunk_ids[idx] for idx in _top_k_by_dot_product(doc_vectors, query_vector, top_k)}
        for query_vector in query_vectors
    ]

    m_values = _read_int_list("ANN_TUNE_M_VALUES", [16, 32])
    ef_construction_values = _read_int_list("ANN_TUNE_EF_CONSTRUCTION_VALUES", [64, 128])
    ef_search_values = _read_int_list("ANN_TUNE_EF_SEARCH_VALUES", [32, 64, 128])

    doc_array = np.array(doc_vectors, dtype="float32")
    query_array = np.array(query_vectors, dtype="float32")
    rows: list[AnnTuningRow] = []
    for m in m_values:
        for ef_construction in ef_construction_values:
            index_hnsw = faiss.IndexHNSWFlat(doc_array.shape[1], m)
            index_hnsw.hnsw.efConstruction = ef_construction
            index_hnsw.add(doc_array)
            for ef_search in ef_search_values:
                index_hnsw.hnsw.efSearch = ef_search
                latencies: list[float] = []
                recalls: list[float] = []
                for query_vector, relevant_ids in zip(query_array, ground_truth):
                    start = time.perf_counter()
                    _, indices = index_hnsw.search(np.array([query_vector], dtype="float32"), top_k)
                    elapsed = (time.perf_counter() - start) * 1000
                    latencies.append(elapsed)
                    predicted_ids = [chunk_ids[idx] for idx in indices[0] if 0 <= idx < len(chunk_ids)]
                    recalls.append(compute_recall_at_k(relevant_ids, predicted_ids))
                rows.append(
                    AnnTuningRow(
                        m=m,
                        ef_construction=ef_construction,
                        ef_search=ef_search,
                        avg_recall_at_k=round(sum(recalls) / len(recalls), 4),
                        avg_latency_ms=round(sum(latencies) / len(latencies), 4),
                        p95_latency_ms=_percentile(latencies, 0.95),
                    )
                )

    report = {
        "status": "ok",
        "cases": len(cases),
        "top_k": top_k,
        "rows": [asdict(row) for row in rows],
        "curve": summarize_curve(rows),
    }
    output = Path(os.getenv("ANN_TUNE_REPORT_PATH", "reports/ann_tuning_report.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from fastapi import FastAPI

from app.contracts import RerankRequest, RerankResponse
from app.services.reranker import SimpleReranker


app = FastAPI(title="rerank-service")
reranker = SimpleReranker()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "rerank-service"}


@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest) -> RerankResponse:
    ranked = reranker.rerank(request.query, request.chunks, top_k=request.top_k)
    return RerankResponse(chunks=ranked)


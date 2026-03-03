from __future__ import annotations

from fastapi import FastAPI

from app.config import get_settings
from app.contracts import RetrievalChunk, RetrievalRequest, RetrievalResponse
from app.services.store import DocumentStore


settings = get_settings()
app = FastAPI(title="retrieval-service")
store = DocumentStore(settings.docs_dir)


@app.on_event("startup")
def startup() -> None:
    store.load()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "retrieval-service"}


@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    chunks = store.search(request.query, top_k=request.top_k)
    return RetrievalResponse(
        chunks=[
            RetrievalChunk(
                chunk_id=item.chunk_id,
                title=item.title,
                url=item.url,
                text=item.text,
                score=item.score,
                bm25_score=item.bm25_score,
                vector_score=item.vector_score,
                keyword_score=item.keyword_score,
            )
            for item in chunks
        ]
    )


@app.post("/reindex")
def reindex() -> dict[str, int]:
    store.load()
    return {"chunks": len(store.chunks)}

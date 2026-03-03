from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.contracts import RagQueryRequest, RagQueryResponse
from app.rag.service import RagGraphService


settings = get_settings()
rag_service = RagGraphService(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    rag_service.startup()
    yield


app = FastAPI(title="rag-agent-service", lifespan=lifespan)


@app.get("/rag/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "rag-agent-service"}


@app.post("/rag/query", response_model=RagQueryResponse)
def rag_query(request: RagQueryRequest) -> RagQueryResponse:
    return rag_service.run(
        session_id=request.session_id,
        query=request.query,
        top_k=request.top_k,
        debug=request.debug,
    )


@app.post("/rag/reindex")
def reindex() -> dict:
    return rag_service.reindex()


@app.get("/rag/stats")
def stats() -> dict:
    return rag_service.stats()

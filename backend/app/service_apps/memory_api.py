from __future__ import annotations

from fastapi import FastAPI

from app.contracts import MemoryQuery, MemoryReadResponse, MemoryWriteRequest
from app.services.memory import MemoryManager


app = FastAPI(title="memory-service")
memory = MemoryManager()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "memory-service"}


@app.post("/memory/write")
def write_memory(request: MemoryWriteRequest) -> dict[str, str]:
    memory.write(request.session_id, request.entry)
    return {"status": "ok"}


@app.post("/memory/read", response_model=MemoryReadResponse)
def read_memory(request: MemoryQuery, kind: str = "short") -> MemoryReadResponse:
    rows = memory.read(request.session_id, kind=kind, key=request.key)
    return MemoryReadResponse(entries=rows)


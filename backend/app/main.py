from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Iterator

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.models import (
    ChatCreateResponse,
    ChatRequest,
    ChatStreamDone,
    FeatureMeta,
    HealthDependency,
    HealthResponse,
    SavedSkill,
)
from app.services.feature_registry import feature_catalog
from app.services.gateway import GatewayDependencies, GatewayOrchestrator
from app.services.service_client import ServiceClient
from app.services.store import DocumentStore
from app.state import ServiceContainer


settings = get_settings()
app = FastAPI(title=settings.app_name)
container = ServiceContainer()
store = DocumentStore(settings.docs_dir)
service_client = ServiceClient(settings, store=store)
gateway = GatewayOrchestrator(
    GatewayDependencies(
        container=container,
        services=service_client,
    )
)


@app.on_event("startup")
def on_startup() -> None:
    store.load()


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    snapshots = container.isolation.snapshot()
    deps = [
        HealthDependency(
            name=name,
            healthy=state.last_error is None,
            circuit_open=state.open_until > time.time(),
            last_error=state.last_error,
            updated_at=state.updated_at,
        )
        for name, state in snapshots.items()
    ]
    overall = all(not dep.circuit_open for dep in deps) if deps else True
    return HealthResponse(app=settings.app_name, healthy=overall, dependencies=deps)


@app.get("/healthz/dependencies", response_model=HealthResponse)
def healthz_dependencies() -> HealthResponse:
    return healthz()


@app.get("/api/features", response_model=list[FeatureMeta])
def get_features() -> list[FeatureMeta]:
    return feature_catalog()


@app.get("/api/skills/saved", response_model=list[SavedSkill])
def get_saved_skills() -> list[SavedSkill]:
    rows = service_client.list_saved_skills().skills
    return [
        SavedSkill(
            id=item.id,
            label=f"{item.name} v{item.version}",
            description=item.description,
        )
        for item in rows
    ]


@app.get("/api/tools")
def get_tools_compat() -> list[dict[str, str]]:
    # Compatibility endpoint kept for previous frontend version.
    return [{"id": item.id, "label": item.label} for item in feature_catalog()]


@app.post("/api/skills/save")
def save_skill(name: str, workflow: str):
    return service_client.save_skill(name=name, workflow=workflow)


@app.post("/api/chat", response_model=ChatCreateResponse)
def create_chat(request: ChatRequest, x_fail_features: str | None = Header(default=None)) -> ChatCreateResponse:
    fail_features = {item.strip() for item in (x_fail_features or "").split(",") if item.strip()}
    response = gateway.create_chat(request, fail_features=fail_features)
    return response


def _sse_stream(text: str, chunk_size: int) -> Iterator[str]:
    for idx in range(0, len(text), chunk_size):
        delta = text[idx : idx + chunk_size]
        yield f"event: message\ndata: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
        time.sleep(0.01)


@app.get("/api/chat/stream")
def stream_chat(session_id: str):
    session = container.session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    def event_iter() -> Iterator[str]:
        yield from _sse_stream(session.text, settings.stream_chunk_size)
        done = ChatStreamDone(
            finish_reason=session.finish_reason,
            status=session.status,
            degraded_features=session.degraded_features,
            sources=session.sources,
            trace_id=session.trace_id,
        )
        yield f"event: done\ndata: {done.model_dump_json()}\n\n"

    return StreamingResponse(event_iter(), media_type="text/event-stream")


@app.get("/api/admin/metrics")
def metrics() -> dict[str, object]:
    snapshots = container.isolation.snapshot()
    return {
        "app": settings.app_name,
        "now": datetime.utcnow().isoformat(),
        "sessions": len(container.session_store._sessions),  # noqa: SLF001
        "dependencies": {
            name: {
                "failures": state.failures,
                "circuit_open": state.open_until > time.time(),
                "last_error": state.last_error,
            }
            for name, state in snapshots.items()
        },
    }

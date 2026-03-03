from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI


app = FastAPI(title="observability-service")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "observability-service"}


@app.get("/metrics")
def metrics() -> dict[str, str]:
    return {
        "service": "observability-service",
        "timestamp": datetime.utcnow().isoformat(),
        "hint": "connect this endpoint to Prometheus or OTEL collector in production",
    }


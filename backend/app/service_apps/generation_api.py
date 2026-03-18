from __future__ import annotations

from fastapi import FastAPI

from app.config import get_settings
from app.contracts import GenerationRequest, GenerationResponse
from app.services.llm import GenerationService


settings = get_settings()
app = FastAPI(title="generation-service")
generator = GenerationService(settings)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "generation-service"}


@app.post("/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest) -> GenerationResponse:
    return generator.generate(
        user_query=request.user_query,
        context_blocks=request.context_blocks,
        feature_notes=request.feature_notes,
        model=request.model,
        temperature=request.temperature,
        top_p=request.top_p,
    )

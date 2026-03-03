from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT_DIR / "docs" / "zyit"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "admissions-gateway"
    service_call_mode: str = "local"
    use_mock_generation: bool = True
    api_url: str = Field(
        default="https://www.right.codes/codex/v1/chat/completions",
        alias="API_URL",
    )
    api_key: str = Field(default="", alias="API_KEY")
    request_timeout_seconds: float = 6.0
    stream_chunk_size: int = 24
    docs_dir: Path = DOCS_DIR

    retrieval_service_url: str = "http://retrieval-service:8001"
    rerank_service_url: str = "http://rerank-service:8002"
    memory_service_url: str = "http://memory-service:8003"
    skill_service_url: str = "http://skill-service:8004"
    generation_service_url: str = "http://generation-service:8005"
    observability_service_url: str = "http://observability-service:8006"

    retrieval_service_timeout_seconds: float = 1.5
    rerank_service_timeout_seconds: float = 1.0
    memory_service_timeout_seconds: float = 0.8
    skill_service_timeout_seconds: float = 1.0
    saved_skill_service_timeout_seconds: float = 1.0
    citation_guard_timeout_seconds: float = 0.4
    generation_service_timeout_seconds: float = 7.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

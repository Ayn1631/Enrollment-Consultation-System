from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT_DIR / "docs" / "zyit"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="admissions-gateway", alias="APP_NAME")
    service_call_mode: str = Field(default="local", alias="SERVICE_CALL_MODE")
    use_mock_generation: bool = Field(default=True, alias="USE_MOCK_GENERATION")
    api_url: str = Field(
        default="https://www.right.codes/codex/v1/chat/completions",
        alias="API_URL",
    )
    api_key: str = Field(default="", alias="API_KEY")
    admin_api_token: str = Field(default="", alias="ADMIN_API_TOKEN")
    rag_stack: str = Field(default="langchain", alias="RAG_STACK")
    agent_stack: str = Field(default="langgraph", alias="AGENT_STACK")
    request_timeout_seconds: float = Field(default=6.0, alias="REQUEST_TIMEOUT_SECONDS")
    stream_chunk_size: int = Field(default=24, alias="STREAM_CHUNK_SIZE")
    docs_dir: Path = DOCS_DIR

    neo4j_uri: str = Field(default="", alias="NEO4J_URI")
    neo4j_user: str = Field(default="", alias="NEO4J_USER")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    langchain4j_service_url: str = Field(default="", alias="LANGCHAIN4J_SERVICE_URL")
    langchain4j_timeout_seconds: float = Field(default=1.5, alias="LANGCHAIN4J_TIMEOUT_SECONDS")

    retrieval_service_url: str = Field(default="http://retrieval-service:8001", alias="RETRIEVAL_SERVICE_URL")
    rerank_service_url: str = Field(default="http://rerank-service:8002", alias="RERANK_SERVICE_URL")
    memory_service_url: str = Field(default="http://memory-service:8003", alias="MEMORY_SERVICE_URL")
    skill_service_url: str = Field(default="http://skill-service:8004", alias="SKILL_SERVICE_URL")
    generation_service_url: str = Field(default="http://generation-service:8005", alias="GENERATION_SERVICE_URL")
    observability_service_url: str = Field(
        default="http://observability-service:8006",
        alias="OBSERVABILITY_SERVICE_URL",
    )

    retrieval_service_timeout_seconds: float = Field(default=1.5, alias="RETRIEVAL_SERVICE_TIMEOUT_SECONDS")
    rerank_service_timeout_seconds: float = Field(default=1.0, alias="RERANK_SERVICE_TIMEOUT_SECONDS")
    memory_service_timeout_seconds: float = Field(default=0.8, alias="MEMORY_SERVICE_TIMEOUT_SECONDS")
    skill_service_timeout_seconds: float = Field(default=1.0, alias="SKILL_SERVICE_TIMEOUT_SECONDS")
    saved_skill_service_timeout_seconds: float = Field(default=1.0, alias="SAVED_SKILL_SERVICE_TIMEOUT_SECONDS")
    citation_guard_timeout_seconds: float = Field(default=0.4, alias="CITATION_GUARD_TIMEOUT_SECONDS")
    generation_service_timeout_seconds: float = Field(default=7.0, alias="GENERATION_SERVICE_TIMEOUT_SECONDS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

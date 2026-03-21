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
    agent_stack: str = Field(default="langgraph", alias="AGENT_STACK")
    request_timeout_seconds: float = Field(default=6.0, alias="REQUEST_TIMEOUT_SECONDS")
    stream_chunk_size: int = Field(default=24, alias="STREAM_CHUNK_SIZE")
    generation_light_model: str = Field(default="gpt-4o-mini", alias="GENERATION_LIGHT_MODEL")
    generation_main_model: str = Field(default="gpt-4.1", alias="GENERATION_MAIN_MODEL")
    generation_cache_enabled: bool = Field(default=True, alias="GENERATION_CACHE_ENABLED")
    generation_cache_ttl_seconds: int = Field(default=300, alias="GENERATION_CACHE_TTL_SECONDS")
    cors_allow_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
        alias="CORS_ALLOW_ORIGINS",
    )
    docs_dir: Path = DOCS_DIR

    embedding_api_url: str = Field(default="", alias="EMBEDDING_API_URL")
    embedding_model: str = Field(default="text-embedding-3-large", alias="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(default=16, alias="EMBEDDING_BATCH_SIZE")
    rag_faiss_dir: Path = Field(default=ROOT_DIR / "backend" / "data" / "faiss", alias="RAG_FAISS_DIR")
    rag_chunk_size: int = Field(default=500, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=80, alias="RAG_CHUNK_OVERLAP")
    rag_retrieve_top_n: int = Field(default=40, alias="RAG_RETRIEVE_TOP_N")
    rag_final_top_k: int = Field(default=8, alias="RAG_FINAL_TOP_K")
    rag_retry_top_n: int = Field(default=64, alias="RAG_RETRY_TOP_N")
    rag_citation_min_sources: int = Field(default=2, alias="RAG_CITATION_MIN_SOURCES")
    rag_citation_min_top1_score: float = Field(default=0.18, alias="RAG_CITATION_MIN_TOP1_SCORE")
    rag_quality_min_coverage: float = Field(default=0.25, alias="RAG_QUALITY_MIN_COVERAGE")
    rag_node_timeout_ms: int = Field(default=1200, alias="RAG_NODE_TIMEOUT_MS")

    neo4j_uri: str = Field(default="", alias="NEO4J_URI")
    neo4j_user: str = Field(default="", alias="NEO4J_USER")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    langchain4j_service_url: str = Field(default="", alias="LANGCHAIN4J_SERVICE_URL")
    langchain4j_timeout_seconds: float = Field(default=1.5, alias="LANGCHAIN4J_TIMEOUT_SECONDS")

    rag_agent_service_url: str = Field(default="http://rag-agent-service:8001", alias="RAG_AGENT_SERVICE_URL")
    memory_service_url: str = Field(default="http://memory-service:8003", alias="MEMORY_SERVICE_URL")
    skill_service_url: str = Field(default="http://skill-service:8004", alias="SKILL_SERVICE_URL")
    generation_service_url: str = Field(default="http://generation-service:8005", alias="GENERATION_SERVICE_URL")
    observability_service_url: str = Field(
        default="http://observability-service:8006",
        alias="OBSERVABILITY_SERVICE_URL",
    )

    rag_agent_service_timeout_seconds: float = Field(default=2.5, alias="RAG_AGENT_SERVICE_TIMEOUT_SECONDS")
    memory_service_timeout_seconds: float = Field(default=0.8, alias="MEMORY_SERVICE_TIMEOUT_SECONDS")
    skill_service_timeout_seconds: float = Field(default=1.0, alias="SKILL_SERVICE_TIMEOUT_SECONDS")
    saved_skill_service_timeout_seconds: float = Field(default=1.0, alias="SAVED_SKILL_SERVICE_TIMEOUT_SECONDS")
    citation_guard_timeout_seconds: float = Field(default=0.4, alias="CITATION_GUARD_TIMEOUT_SECONDS")
    generation_service_timeout_seconds: float = Field(default=7.0, alias="GENERATION_SERVICE_TIMEOUT_SECONDS")

    def resolve_embedding_api_url(self) -> str:
        """从 API_URL 推导 Embedding 端点，支持同源 OpenAI 兼容接口。"""
        if self.embedding_api_url:
            return self.embedding_api_url.strip()
        base = self.api_url.strip()
        if base.endswith("/chat/completions"):
            return f"{base[:-len('/chat/completions')]}/embeddings"
        return f"{base.rstrip('/')}/embeddings"

    def resolve_cors_allow_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_allow_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

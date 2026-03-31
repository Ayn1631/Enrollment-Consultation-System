from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT_DIR / "docs" / "zyit"
logger = logging.getLogger(__name__)


def _normalize_openai_base_url(endpoint: str) -> str:
    normalized = (endpoint or "").strip()
    for suffix in ("/chat/completions", "/responses", "/completions"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized.rstrip("/")


def _is_local_endpoint(endpoint: str) -> bool:
    hostname = (urlparse(endpoint).hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "::1"}


@lru_cache(maxsize=8)
def _llm_endpoint_reachable(endpoint: str, api_key: str) -> bool:
    base_url = _normalize_openai_base_url(endpoint)
    if not base_url:
        return False
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.get(f"{base_url}/models", headers=headers, timeout=1.5)
        return response.status_code < 500
    except Exception:
        return False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / "backend" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = Field(default="admissions-gateway", alias="APP_NAME")
    service_call_mode: str = Field(default="local", alias="SERVICE_CALL_MODE")
    use_mock_generation: bool = Field(default=True, alias="USE_MOCK_GENERATION")
    api_url: str = Field(
        default="https://www.right.codes/codex/v1/chat/completions",
        alias="API_URL",
    )
    api_key: str = Field(default="", alias="API_KEY")
    llm_api_url: str = Field(default="", alias="LLM_API_URL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    admin_api_token: str = Field(default="", alias="ADMIN_API_TOKEN")
    agent_stack: str = Field(default="langgraph", alias="AGENT_STACK")
    request_timeout_seconds: float = Field(default=6.0, alias="REQUEST_TIMEOUT_SECONDS")
    llm_timeout_seconds: float = Field(default=20.0, alias="LLM_TIMEOUT_SECONDS")
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
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="BAAI/bge-large-zh-v1.5", alias="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(default=16, alias="EMBEDDING_BATCH_SIZE")
    rerank_api_url: str = Field(default="", alias="RERANK_API_URL")
    rerank_api_key: str = Field(default="", alias="RERANK_API_KEY")
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANK_MODEL")
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
    mcp_enabled: bool = Field(default=True, alias="MCP_ENABLED")
    mcp_config_path: str = Field(default="", alias="MCP_CONFIG_PATH")

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

    def resolve_llm_api_url(self) -> str:
        """解析对话模型地址，优先使用专用 LLM_API_URL，未配置时回退 API_URL。"""
        primary = (self.llm_api_url or "").strip()
        fallback = (self.api_url or "").strip()
        if primary and fallback and primary != fallback and _is_local_endpoint(primary):
            api_key = (self.api_key or self.llm_api_key or "").strip()
            if not _llm_endpoint_reachable(primary, api_key):
                logger.warning("本地 LLM_API_URL 不可达，已回退到 API_URL: %s -> %s", primary, fallback)
                return fallback
        return primary or fallback

    def resolve_llm_api_key(self) -> str:
        """解析对话模型密钥，优先使用专用 LLM_API_KEY，未配置时回退 API_KEY。"""
        primary_url = (self.llm_api_url or "").strip()
        fallback_url = (self.api_url or "").strip()
        primary_key = (self.llm_api_key or "").strip()
        fallback_key = (self.api_key or "").strip()
        if primary_url and fallback_url and primary_url != fallback_url and _is_local_endpoint(primary_url):
            probe_key = fallback_key or primary_key
            if not _llm_endpoint_reachable(primary_url, probe_key):
                return fallback_key or primary_key
        return primary_key or fallback_key

    def resolve_embedding_api_url(self) -> str:
        """解析 Embedding 端点，未单独配置时从对话端点推导。"""
        if self.embedding_api_url:
            return self.embedding_api_url.strip()
        base = self.resolve_llm_api_url()
        if base.endswith("/chat/completions"):
            return f"{base[:-len('/chat/completions')]}/embeddings"
        return f"{base.rstrip('/')}/embeddings"

    def resolve_embedding_api_key(self) -> str:
        """解析 Embedding 密钥，未单独配置时回退到对话模型密钥。"""
        return (self.embedding_api_key or self.resolve_llm_api_key()).strip()

    def resolve_rerank_api_url(self) -> str:
        """解析 Rerank 端点，未单独配置时从对话端点推导。"""
        if self.rerank_api_url:
            return self.rerank_api_url.strip()
        base = self.resolve_llm_api_url()
        if base.endswith("/chat/completions"):
            return f"{base[:-len('/chat/completions')]}/rerank"
        return f"{base.rstrip('/')}/rerank"

    def resolve_rerank_api_key(self) -> str:
        """解析 Rerank 密钥，未单独配置时回退到对话模型密钥。"""
        return (self.rerank_api_key or self.resolve_llm_api_key()).strip()

    def resolve_cors_allow_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_allow_origins.split(",") if item.strip()]

    def resolve_mcp_config_path(self) -> Path | None:
        """解析 MCP 配置文件路径，优先使用显式环境变量。"""
        candidate = self.mcp_config_path.strip()
        if candidate:
            return Path(candidate).expanduser()
        fallback_candidates = [
            ROOT_DIR / "backend" / "config" / "mcp.json",
            ROOT_DIR / "backend" / "config" / "mcp_settings.json",
            ROOT_DIR / "backend" / "config.json",
            ROOT_DIR / "config" / "mcp.json",
            ROOT_DIR / "config.json",
        ]
        for path in fallback_candidates:
            if path.exists():
                return path
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

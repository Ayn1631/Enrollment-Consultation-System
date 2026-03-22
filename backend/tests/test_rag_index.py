from __future__ import annotations

import httpx
import math

from app.config import Settings
from app.rag.index import OpenAICompatibleEmbeddings


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_embed_documents_batches_remote_requests(monkeypatch, runtime_settings):
    captured_batches: list[list[str]] = []
    captured_payloads: list[dict] = []
    captured_headers: list[dict] = []
    captured_urls: list[str] = []

    class _FakeClient:
        def __init__(self, timeout: float):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, headers: dict | None = None, json: dict | None = None):
            captured_urls.append(url)
            captured_headers.append(headers or {})
            captured_payloads.append(dict(json or {}))
            texts = list((json or {}).get("input", []))
            captured_batches.append(texts)
            return _FakeResponse(
                {
                    "data": [
                        {"embedding": [float(idx + 1), 0.0, 0.0]}
                        for idx, _ in enumerate(texts)
                    ]
                }
            )

    monkeypatch.setattr("app.rag.index.httpx.Client", _FakeClient)
    if not runtime_settings.resolve_embedding_api_key():
        monkeypatch.setenv("EMBEDDING_API_KEY", "runtime-embedding-key")
        runtime_settings = Settings()
    embeddings = OpenAICompatibleEmbeddings(
        endpoint=runtime_settings.resolve_embedding_api_url(),
        api_key=runtime_settings.resolve_embedding_api_key(),
        model=runtime_settings.embedding_model,
        timeout_seconds=runtime_settings.request_timeout_seconds,
        batch_size=2,
        force_local=False,
    )

    rows = embeddings.embed_documents(["a", "b", "c", "d", "e"])

    assert len(rows) == 5
    assert captured_batches == [["a", "b"], ["c", "d"], ["e"]]
    assert all(url == runtime_settings.resolve_embedding_api_url() for url in captured_urls)
    assert all(payload["model"] == runtime_settings.embedding_model for payload in captured_payloads)
    assert all(headers["Authorization"] == f"Bearer {runtime_settings.resolve_embedding_api_key()}" for headers in captured_headers)


def test_embed_documents_falls_back_to_local_vectors_when_remote_batch_fails(monkeypatch, runtime_settings):
    class _FailingClient:
        def __init__(self, timeout: float):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, headers: dict | None = None, json: dict | None = None):
            raise httpx.ConnectError("embedding timeout")

    monkeypatch.setattr("app.rag.index.httpx.Client", _FailingClient)
    if not runtime_settings.resolve_embedding_api_key():
        monkeypatch.setenv("EMBEDDING_API_KEY", "runtime-embedding-key")
        runtime_settings = Settings()
    embeddings = OpenAICompatibleEmbeddings(
        endpoint=runtime_settings.resolve_embedding_api_url(),
        api_key=runtime_settings.resolve_embedding_api_key(),
        model=runtime_settings.embedding_model,
        timeout_seconds=runtime_settings.request_timeout_seconds,
        batch_size=2,
        force_local=False,
    )

    rows = embeddings.embed_documents(["招生政策", "学费标准"])

    assert len(rows) == 2
    assert all(len(vector) == 32 for vector in rows)


def test_settings_resolve_embedding_and_rerank_api_urls():
    settings = Settings()

    assert settings.resolve_embedding_api_url()
    assert settings.resolve_rerank_api_url()
    assert settings.resolve_llm_api_url()
    if settings.resolve_llm_api_key():
        assert settings.resolve_embedding_api_key()
        assert settings.resolve_rerank_api_key()


def test_settings_prefer_split_model_endpoints_and_keys(monkeypatch):
    monkeypatch.setenv("API_URL", "https://fallback.example/v1/chat/completions")
    monkeypatch.setenv("API_KEY", "primary-env-key")
    monkeypatch.setenv("LLM_API_URL", "https://llm.example/v1/chat/completions")
    monkeypatch.setenv("LLM_API_KEY", "llm-env-key")
    monkeypatch.setenv("EMBEDDING_API_URL", "https://embed.example/v1/embeddings")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-env-key")
    monkeypatch.setenv("RERANK_API_URL", "https://rerank.example/v1/rerank")
    monkeypatch.setenv("RERANK_API_KEY", "rerank-env-key")
    settings = Settings()

    assert settings.resolve_llm_api_url() == "https://llm.example/v1/chat/completions"
    assert settings.resolve_llm_api_key() == "llm-env-key"
    assert settings.resolve_embedding_api_url() == "https://embed.example/v1/embeddings"
    assert settings.resolve_embedding_api_key() == "embed-env-key"
    assert settings.resolve_rerank_api_url() == "https://rerank.example/v1/rerank"
    assert settings.resolve_rerank_api_key() == "rerank-env-key"


def test_embedding_result_for_real_input_text_is_stable_and_normalized(runtime_settings):
    from app.rag.service import RagGraphService

    if not runtime_settings.docs_dir.exists():
        raise AssertionError(f"真实语料目录不存在: {runtime_settings.docs_dir}")
    settings = runtime_settings.model_copy(update={"rag_faiss_dir": runtime_settings.rag_faiss_dir})
    service = RagGraphService(settings)
    service.index.reindex()
    docs = service.index.all_documents()
    target = next(
        (
            doc
            for doc in docs
            if doc.metadata.get("chunk_level") == "small"
            and str(doc.metadata.get("chunk_text", "")).strip()
            and doc.metadata.get("query_expansions")
        ),
        None,
    )
    assert target is not None
    text = str(target.metadata.get("chunk_text", "")).strip()
    embeddings = OpenAICompatibleEmbeddings(
        endpoint=settings.resolve_embedding_api_url(),
        api_key=settings.resolve_embedding_api_key(),
        model=settings.embedding_model,
        timeout_seconds=settings.request_timeout_seconds,
        batch_size=settings.embedding_batch_size,
        force_local=settings.use_mock_generation,
    )

    query_vector = embeddings.embed_query(text)
    doc_vectors = embeddings.embed_documents([text, f"{text} 补充说明"])

    assert query_vector
    assert len(doc_vectors) == 2
    assert len(query_vector) == len(doc_vectors[0])
    assert all(math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9) for left, right in zip(query_vector, doc_vectors[0]))
    assert any(not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9) for left, right in zip(query_vector, doc_vectors[1]))
    norm = math.sqrt(sum(item * item for item in query_vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6)

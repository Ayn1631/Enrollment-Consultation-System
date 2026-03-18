from __future__ import annotations

import httpx

from app.rag.index import OpenAICompatibleEmbeddings


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_embed_documents_batches_remote_requests(monkeypatch):
    captured_batches: list[list[str]] = []

    class _FakeClient:
        def __init__(self, timeout: float):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, headers: dict | None = None, json: dict | None = None):
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
    embeddings = OpenAICompatibleEmbeddings(
        endpoint="https://example.com/embeddings",
        api_key="test-key",
        model="text-embedding-3-large",
        timeout_seconds=2.0,
        batch_size=2,
        force_local=False,
    )

    rows = embeddings.embed_documents(["a", "b", "c", "d", "e"])

    assert len(rows) == 5
    assert captured_batches == [["a", "b"], ["c", "d"], ["e"]]


def test_embed_documents_falls_back_to_local_vectors_when_remote_batch_fails(monkeypatch):
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
    embeddings = OpenAICompatibleEmbeddings(
        endpoint="https://example.com/embeddings",
        api_key="test-key",
        model="text-embedding-3-large",
        timeout_seconds=2.0,
        batch_size=2,
        force_local=False,
    )

    rows = embeddings.embed_documents(["招生政策", "学费标准"])

    assert len(rows) == 2
    assert all(len(vector) == 32 for vector in rows)

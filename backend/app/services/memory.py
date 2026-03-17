from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock

from app.contracts import MemoryEntry


@dataclass
class SessionMemory:
    short: dict[str, MemoryEntry] = field(default_factory=dict)
    long: dict[str, MemoryEntry] = field(default_factory=dict)
    special: dict[str, MemoryEntry] = field(default_factory=dict)


class MemoryManager:
    def __init__(self):
        self._store: dict[str, SessionMemory] = {}
        self._lock = Lock()

    def _get(self, session_id: str) -> SessionMemory:
        return self._store.setdefault(session_id, SessionMemory())

    def write(self, session_id: str, entry: MemoryEntry) -> None:
        if entry.expires_at is None and entry.kind == "short":
            entry.expires_at = datetime.utcnow() + timedelta(hours=4)
        if entry.expires_at is None and entry.kind == "special":
            entry.expires_at = datetime.utcnow() + timedelta(days=90)
        with self._lock:
            session = self._get(session_id)
            bucket = self._bucket(session, entry.kind)
            bucket[entry.key] = entry

    def append_long_summary(self, session_id: str, snippet: str, max_length: int = 600) -> MemoryEntry:
        """维护滚动长期摘要，避免每轮都做全量重写。"""
        with self._lock:
            session = self._get(session_id)
            bucket = self._bucket(session, "long")
            existing = bucket.get("rolling_summary")
            previous = existing.value if existing else ""
            merged = self._merge_summary(previous=previous, snippet=snippet, max_length=max_length)
            entry = MemoryEntry(
                key="rolling_summary",
                value=merged,
                kind="long",
                confidence=0.72,
                source="rolling_summary",
                last_verified=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=30),
            )
            bucket[entry.key] = entry
            return entry

    def read(self, session_id: str, kind: str, key: str | None = None) -> list[MemoryEntry]:
        with self._lock:
            session = self._get(session_id)
            bucket = self._bucket(session, kind)
            now = datetime.utcnow()
            keys_to_delete = [
                k
                for k, v in bucket.items()
                if v.expires_at is not None and v.expires_at <= now
            ]
            for k in keys_to_delete:
                del bucket[k]
            if key:
                item = bucket.get(key)
                return [item] if item else []
            return sorted(bucket.values(), key=lambda x: x.last_verified, reverse=True)

    def _bucket(self, session: SessionMemory, kind: str) -> dict[str, MemoryEntry]:
        if kind == "short":
            return session.short
        if kind == "long":
            return session.long
        if kind == "special":
            return session.special
        raise ValueError(f"unknown memory kind: {kind}")

    def _merge_summary(self, previous: str, snippet: str, max_length: int) -> str:
        """增量合并长期摘要，保持信息简洁。"""
        normalized_prev = " ".join(previous.split()).strip()
        normalized_snippet = " ".join(snippet.split()).strip()
        if not normalized_prev:
            return normalized_snippet[:max_length]
        if normalized_snippet in normalized_prev:
            return normalized_prev[:max_length]
        merged = f"{normalized_prev} | {normalized_snippet}".strip(" |")
        if len(merged) <= max_length:
            return merged
        return merged[-max_length:]

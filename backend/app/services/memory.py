from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

from app.contracts import MemoryEntry


DEFAULT_MEMORY_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "memory" / "memory.sqlite3"


class MemoryManager:
    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path) if db_path is not None else DEFAULT_MEMORY_DB_PATH
        self._lock = Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def write(self, session_id: str, entry: MemoryEntry) -> None:
        normalized = self._normalize_entry(entry)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_entries (
                    session_id, kind, key, value, confidence, source, last_verified, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, kind, key) DO UPDATE SET
                    value = excluded.value,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    last_verified = excluded.last_verified,
                    expires_at = excluded.expires_at
                """,
                (
                    session_id,
                    normalized.kind,
                    normalized.key,
                    normalized.value,
                    normalized.confidence,
                    normalized.source,
                    self._serialize_dt(normalized.last_verified),
                    self._serialize_dt(normalized.expires_at),
                ),
            )
            conn.commit()

    def append_long_summary(self, session_id: str, snippet: str, max_length: int = 600) -> MemoryEntry:
        with self._lock, self._connect() as conn:
            existing = self._read_entries(conn=conn, session_id=session_id, kind="long", key="rolling_summary")
            previous = existing[0].value if existing else ""
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
            normalized = self._normalize_entry(entry)
            conn.execute(
                """
                INSERT INTO memory_entries (
                    session_id, kind, key, value, confidence, source, last_verified, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, kind, key) DO UPDATE SET
                    value = excluded.value,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    last_verified = excluded.last_verified,
                    expires_at = excluded.expires_at
                """,
                (
                    session_id,
                    normalized.kind,
                    normalized.key,
                    normalized.value,
                    normalized.confidence,
                    normalized.source,
                    self._serialize_dt(normalized.last_verified),
                    self._serialize_dt(normalized.expires_at),
                ),
            )
            conn.commit()
            return normalized

    def read(self, session_id: str, kind: str, key: str | None = None) -> list[MemoryEntry]:
        with self._lock, self._connect() as conn:
            return self._read_entries(conn=conn, session_id=session_id, kind=kind, key=key)

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    last_verified TEXT NOT NULL,
                    expires_at TEXT,
                    PRIMARY KEY (session_id, kind, key)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_session_kind_verified
                ON memory_entries(session_id, kind, last_verified DESC)
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _read_entries(
        self,
        *,
        conn: sqlite3.Connection,
        session_id: str,
        kind: str,
        key: str | None = None,
    ) -> list[MemoryEntry]:
        self._purge_expired(conn)
        if key:
            rows = conn.execute(
                """
                SELECT key, value, kind, confidence, source, last_verified, expires_at
                FROM memory_entries
                WHERE session_id = ? AND kind = ? AND key = ?
                ORDER BY last_verified DESC
                """,
                (session_id, kind, key),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT key, value, kind, confidence, source, last_verified, expires_at
                FROM memory_entries
                WHERE session_id = ? AND kind = ?
                ORDER BY last_verified DESC
                """,
                (session_id, kind),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def _purge_expired(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "DELETE FROM memory_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (self._serialize_dt(datetime.utcnow()),),
        )
        conn.commit()

    def _normalize_entry(self, entry: MemoryEntry) -> MemoryEntry:
        normalized = entry.model_copy(deep=True)
        if normalized.expires_at is None and normalized.kind == "short":
            normalized.expires_at = datetime.utcnow() + timedelta(hours=4)
        if normalized.expires_at is None and normalized.kind == "special":
            normalized.expires_at = datetime.utcnow() + timedelta(days=90)
        return normalized

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            key=str(row["key"]),
            value=str(row["value"]),
            kind=str(row["kind"]),
            confidence=float(row["confidence"]),
            source=str(row["source"]),
            last_verified=self._deserialize_dt(row["last_verified"]) or datetime.utcnow(),
            expires_at=self._deserialize_dt(row["expires_at"]),
        )

    def _serialize_dt(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.isoformat(timespec="microseconds")

    def _deserialize_dt(self, value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value)

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

"""SQLite-backed checkpoint store providing idempotent, resumable progress.

Design:
    * The store holds the **embeddings themselves**, not just "done" flags. Each successful
      batch is committed in a single transaction, so a chunk is either fully persisted with
      its vector or not at all. The final Parquet file is *exported* from this store, so a
      crash can never leave a half-written output or lose completed work.
    * Rows are keyed by ``(chunk_id, model_key)``. Inserts use ``INSERT OR IGNORE`` so
      replaying a batch (e.g. after a crash between API success and commit) is harmless.
    * WAL journaling + ``synchronous=NORMAL``: committed transactions survive process
      crashes / Ctrl+C. A power loss may drop the last few commits, which merely causes
      those chunks to be re-embedded on the next run.
    * One connection guarded by a lock, usable from worker threads (``asyncio.to_thread``).
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import NamedTuple, Self

import numpy as np

from src.schemas import EmbeddingResult

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY,
    chunk_id    TEXT    NOT NULL,
    model_key   TEXT    NOT NULL,
    record_id   TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    token_count INTEGER NOT NULL,
    dim         INTEGER NOT NULL,
    embedding   BLOB    NOT NULL,
    created_at  TEXT    NOT NULL,
    UNIQUE (chunk_id, model_key)
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""
_SCHEMA_VERSION = "1"

# Stay well below SQLite's bound-parameter limit (32766 on modern builds).
_MAX_SQL_PARAMS = 900

#: Vectors are stored as little-endian float32 (half the size of float64, ample precision).
VECTOR_DTYPE = np.dtype("<f4")


class StoredEmbedding(NamedTuple):
    """A persisted chunk + vector, as read back from the store."""

    record_id: str
    chunk_id: str
    chunk_index: int
    text: str
    token_count: int
    embedding: np.ndarray


class CheckpointStore:
    """Durable record of which chunks have been embedded (and their vectors)."""

    def __init__(self, path: Path | str) -> None:
        """Open (or create) the checkpoint database.

        Args:
            path: SQLite file path; parent directories are created. ``":memory:"`` is
                accepted for tests.
        """
        self.path = Path(path) if str(path) != ":memory:" else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None, timeout=30.0
        )
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (_SCHEMA_VERSION,),
            )
        logger.debug("checkpoint_opened", extra={"path": str(path)})

    # -- context management ------------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        with self._lock, contextlib.suppress(sqlite3.ProgrammingError):
            self._conn.close()

    # -- queries -----------------------------------------------------------------------

    def pending_chunk_ids(self, chunk_ids: Sequence[str], model_key: str) -> set[str]:
        """Return the subset of ``chunk_ids`` that has *not* been embedded yet.

        Args:
            chunk_ids: Candidate chunk ids.
            model_key: Embedding namespace (see ``EmbeddingSettings.model_key``).

        Returns:
            Ids still requiring an API call.
        """
        if not chunk_ids:
            return set()
        done: set[str] = set()
        with self._lock:
            for offset in range(0, len(chunk_ids), _MAX_SQL_PARAMS):
                window = chunk_ids[offset : offset + _MAX_SQL_PARAMS]
                placeholders = ",".join("?" * len(window))
                rows = self._conn.execute(
                    f"SELECT chunk_id FROM embeddings WHERE model_key = ? "
                    f"AND chunk_id IN ({placeholders})",
                    (model_key, *window),
                ).fetchall()
                done.update(row[0] for row in rows)
        return set(chunk_ids) - done

    def is_completed(self, chunk_id: str, model_key: str) -> bool:
        """Return whether a single chunk has been embedded."""
        return not self.pending_chunk_ids([chunk_id], model_key)

    def completed_count(self, model_key: str) -> int:
        """Number of chunks embedded under ``model_key``."""
        with self._lock:
            (count,) = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_key = ?", (model_key,)
            ).fetchone()
        return int(count)

    def model_keys(self) -> list[str]:
        """All embedding namespaces present in the store."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT model_key FROM embeddings ORDER BY model_key"
            ).fetchall()
        return [row[0] for row in rows]

    # -- writes ------------------------------------------------------------------------

    def save_results(self, results: Sequence[EmbeddingResult], model_key: str) -> int:
        """Atomically persist a batch of embeddings.

        Args:
            results: Embedded chunks from one API call.
            model_key: Embedding namespace.

        Returns:
            Number of newly inserted rows (already-present chunks are ignored).
        """
        if not results:
            return 0
        now = datetime.now(UTC).isoformat()
        rows = [
            (
                result.chunk.chunk_id,
                model_key,
                result.chunk.record_id,
                result.chunk.chunk_index,
                result.chunk.text,
                result.chunk.token_count,
                int(result.embedding.shape[0]),
                np.ascontiguousarray(result.embedding, dtype=VECTOR_DTYPE).tobytes(),
                now,
            )
            for result in results
        ]
        with self._lock:
            before = self._conn.total_changes
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO embeddings (chunk_id, model_key, record_id, "
                    "chunk_index, text, token_count, dim, embedding, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            return self._conn.total_changes - before

    # -- export ------------------------------------------------------------------------

    def iter_results(
        self, model_key: str, batch_size: int = 10_000
    ) -> Iterator[list[StoredEmbedding]]:
        """Stream persisted embeddings in insertion order, ``batch_size`` rows at a time.

        Args:
            model_key: Embedding namespace to export.
            batch_size: Rows per yielded list (bounds memory during export).

        Yields:
            Lists of ``StoredEmbedding``.
        """
        last_id = 0
        while True:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id, record_id, chunk_id, chunk_index, text, token_count, embedding "
                    "FROM embeddings WHERE model_key = ? AND id > ? ORDER BY id LIMIT ?",
                    (model_key, last_id, batch_size),
                ).fetchall()
            if not rows:
                return
            last_id = rows[-1][0]
            yield [
                StoredEmbedding(
                    record_id=record_id,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    text=text,
                    token_count=token_count,
                    embedding=np.frombuffer(blob, dtype=VECTOR_DTYPE),
                )
                for _, record_id, chunk_id, chunk_index, text, token_count, blob in rows
            ]

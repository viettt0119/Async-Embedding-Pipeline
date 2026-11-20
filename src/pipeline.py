"""Orchestrator: CSV -> clean -> chunk -> skip done -> batch -> embed -> checkpoint -> Parquet.

Concurrency model::

    ┌────────────── worker thread ──────────────┐      ┌──────── event loop ────────┐
    │ CsvCleaner → TokenChunker → checkpoint     │      │  N consumer tasks           │
    │ filter (skip done) → BatchBuilder          │ ───▶ │  embed_batch() → save to    │
    └──────── bounded asyncio.Queue (backpressure) ───▶ │  SQLite (to_thread) → tqdm  │
                                                        └─────────────────────────────┘

The producer generator is CPU/IO bound (pandas parsing, tiktoken) and is advanced one batch
at a time through ``asyncio.to_thread`` so it never blocks the event loop that drives HTTP.
A bounded queue gives backpressure: reading never runs far ahead of embedding.

Interruptions (Ctrl+C, crash, network loss) lose at most the in-flight batches; everything
committed is skipped on the next run. After a run, Parquet is exported from the checkpoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from src.checkpoint import CheckpointStore
from src.chunker import BatchBuilder, TokenChunker
from src.cleaner import CsvCleaner
from src.embedder import AsyncEmbedder
from src.schemas import Batch, Chunk, PipelineConfig

logger = logging.getLogger(__name__)

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("chunk_id", pa.string(), nullable=False),
        pa.field("chunk_index", pa.int32(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("token_count", pa.int32(), nullable=False),
        pa.field("embedding", pa.list_(pa.float32()), nullable=False),
    ]
)


@dataclass(slots=True)
class PipelineReport:
    """Summary of a pipeline run."""

    run_id: str
    model_key: str
    rows_read: int = 0
    records_emitted: int = 0
    duplicates_dropped: int = 0
    empty_rows_dropped: int = 0
    chunks_total: int = 0
    chunks_skipped: int = 0
    chunks_embedded: int = 0
    chunks_failed: int = 0
    batches_total: int = 0
    batches_succeeded: int = 0
    batches_failed: int = 0
    api_requests: int = 0
    retries: int = 0
    tokens_used: int = 0
    estimated_tokens: int = 0
    duration_seconds: float = 0.0
    output_path: str | None = None
    rows_exported: int = 0
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dict."""
        return asdict(self)


class EmbeddingPipeline:
    """Coordinates cleaning, chunking, checkpoint filtering, async embedding and export."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        store: CheckpointStore,
        embedder: AsyncEmbedder | None = None,
        cleaner: CsvCleaner | None = None,
        chunker: TokenChunker | None = None,
        run_id: str | None = None,
        show_progress: bool = True,
    ) -> None:
        """Create a pipeline.

        Args:
            config: Validated configuration.
            store: Checkpoint store (owned by the pipeline; closed by ``close()``).
            embedder: Async embedder; may be ``None`` for ``plan()``/``export()`` only.
            cleaner: Optional pre-built cleaner (defaults to one built from config).
            chunker: Optional pre-built chunker (defaults to one built from config).
            run_id: Identifier for logs and dead-letter entries.
            show_progress: Render a tqdm progress bar.
        """
        self.config = config
        self.store = store
        self.embedder = embedder
        self.cleaner = cleaner or CsvCleaner.from_settings(config.data)
        self.chunker = chunker or TokenChunker.from_settings(config.chunking)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.show_progress = show_progress
        self.model_key = config.embedding.model_key
        self._report = PipelineReport(run_id=self.run_id, model_key=self.model_key)

    # -- lifecycle -----------------------------------------------------------------------

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
        """Release the checkpoint store."""
        self.store.close()

    # -- producer (runs in a worker thread) ----------------------------------------------

    def iter_pending_batches(self) -> Iterator[Batch]:
        """Yield batches containing only chunks that are not yet checkpointed.

        Chunks are checked against the store in groups of ``runtime.pending_lookup_size``
        to amortise SQL round-trips while keeping memory bounded.
        """
        builder = BatchBuilder.from_settings(self.config.batching)
        group_size = self.config.runtime.pending_lookup_size
        buffer: list[Chunk] = []

        for record in self.cleaner.iter_records():
            buffer.extend(self.chunker.chunk_record(record))
            if len(buffer) >= group_size:
                yield from self._filter_and_batch(buffer, builder)
                buffer = []
        if buffer:
            yield from self._filter_and_batch(buffer, builder)
        if (last := builder.flush()) is not None:
            self._count_batch(last)
            yield last

    def _filter_and_batch(self, chunks: list[Chunk], builder: BatchBuilder) -> Iterator[Batch]:
        pending = self.store.pending_chunk_ids([chunk.chunk_id for chunk in chunks], self.model_key)
        self._report.chunks_total += len(chunks)
        self._report.chunks_skipped += len(chunks) - len(pending)
        for chunk in chunks:
            if chunk.chunk_id in pending and (batch := builder.add(chunk)) is not None:
                self._count_batch(batch)
                yield batch

    def _count_batch(self, batch: Batch) -> None:
        self._report.batches_total += 1
        self._report.estimated_tokens += batch.total_tokens

    # -- dry run -------------------------------------------------------------------------

    def plan(self) -> PipelineReport:
        """Clean, chunk and batch without calling the API (cost/volume estimate)."""
        started = time.perf_counter()
        for _ in self.iter_pending_batches():
            pass
        self._finalize_report(started)
        self._report.dry_run = True
        return self._report

    # -- async execution -----------------------------------------------------------------

    async def run(self, *, export: bool = True) -> PipelineReport:
        """Execute the pipeline end to end.

        Args:
            export: Write the Parquet output from the checkpoint after embedding.

        Returns:
            The run report.

        Raises:
            FatalEmbeddingError: On credential/quota errors (the run is aborted).
            asyncio.CancelledError: If cancelled (e.g. Ctrl+C); committed work is kept.
        """
        if self.embedder is None:
            raise ValueError("run() requires an embedder; use plan() for a dry run")

        started = time.perf_counter()
        workers = self.config.rate_limit.max_concurrency
        queue: asyncio.Queue[Batch | None] = asyncio.Queue(
            maxsize=workers * self.config.runtime.queue_size_per_worker
        )
        logger.info(
            "pipeline_started",
            extra={
                "run_id": self.run_id,
                "model_key": self.model_key,
                "input_csv": str(self.config.data.input_csv),
                "workers": workers,
                "already_completed": self.store.completed_count(self.model_key),
            },
        )

        progress = tqdm(
            total=0,
            unit="chunk",
            desc="Embedding",
            dynamic_ncols=True,
            disable=not self.show_progress,
        )
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(self._produce(queue, progress, workers), name="producer")
                for index in range(workers):
                    group.create_task(self._consume(queue, progress), name=f"worker-{index}")
        except BaseExceptionGroup as errors:
            # Surface the root cause instead of an opaque group for single failures.
            if len(errors.exceptions) == 1:
                raise errors.exceptions[0] from None
            raise
        except asyncio.CancelledError:
            logger.warning(
                "pipeline_interrupted",
                extra={
                    "run_id": self.run_id,
                    "checkpointed": self.store.completed_count(self.model_key),
                    "hint": "re-run the same command to resume",
                },
            )
            raise
        finally:
            progress.close()

        if export:
            self.export()
        self._finalize_report(started)
        logger.info("pipeline_finished", extra=self._report.as_dict())
        return self._report

    async def _produce(
        self, queue: asyncio.Queue[Batch | None], progress: tqdm, workers: int
    ) -> None:
        batches = self.iter_pending_batches()
        while (batch := await asyncio.to_thread(next, batches, None)) is not None:
            # Total grows as the stream is discovered; ETA becomes exact once reading ends.
            progress.total += batch.size
            progress.refresh()
            await queue.put(batch)
        for _ in range(workers):
            await queue.put(None)

    async def _consume(self, queue: asyncio.Queue[Batch | None], progress: tqdm) -> None:
        assert self.embedder is not None
        stats = self.embedder.stats
        while (batch := await queue.get()) is not None:
            outcome = await self.embedder.embed_batch(batch)
            if outcome.succeeded:
                await asyncio.to_thread(self.store.save_results, outcome.results, self.model_key)
                self._report.chunks_embedded += batch.size
            else:
                self._report.chunks_failed += batch.size
            progress.update(batch.size)
            progress.set_postfix(
                tokens=f"{stats.tokens_used:,}",
                retries=stats.retries,
                failed=stats.failed_batches,
                skipped=self._report.chunks_skipped,
                refresh=False,
            )

    # -- export --------------------------------------------------------------------------

    def export(self, output_path: Path | str | None = None) -> int:
        """Write every checkpointed embedding for the current model to Parquet.

        The file is written to a temporary sibling and atomically renamed, so readers never
        observe a partial file.

        Args:
            output_path: Destination (defaults to ``storage.output_parquet``).

        Returns:
            Number of rows written.
        """
        path = Path(output_path or self.config.storage.output_parquet)
        rows = export_parquet(
            self.store,
            self.model_key,
            path,
            row_group_size=self.config.storage.parquet_row_group_size,
        )
        self._report.output_path = str(path)
        self._report.rows_exported = rows
        return rows

    def _finalize_report(self, started: float) -> None:
        report = self._report
        stats = self.cleaner.stats
        report.rows_read = stats.rows_read
        report.records_emitted = stats.emitted
        report.duplicates_dropped = stats.dropped_duplicates
        report.empty_rows_dropped = stats.dropped_empty + stats.dropped_too_short
        if self.embedder is not None:
            embedder_stats = self.embedder.stats
            report.batches_succeeded = embedder_stats.succeeded_batches
            report.batches_failed = embedder_stats.failed_batches
            report.api_requests = embedder_stats.requests
            report.retries = embedder_stats.retries
            report.tokens_used = embedder_stats.tokens_used
        report.duration_seconds = round(time.perf_counter() - started, 3)


def export_parquet(
    store: CheckpointStore, model_key: str, path: Path, *, row_group_size: int = 10_000
) -> int:
    """Stream embeddings from the checkpoint into a Parquet file (atomic replace).

    Args:
        store: Source checkpoint store.
        model_key: Embedding namespace to export.
        path: Destination ``.parquet`` file.
        row_group_size: Rows per Parquet row group (also the read batch size).

    Returns:
        Number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    schema = PARQUET_SCHEMA.with_metadata({"model_key": model_key})
    written = 0
    with pq.ParquetWriter(tmp_path, schema, compression="zstd") as writer:
        for rows in store.iter_results(model_key, batch_size=row_group_size):
            vectors = [row.embedding for row in rows]
            offsets = np.zeros(len(vectors) + 1, dtype=np.int32)
            np.cumsum([vector.shape[0] for vector in vectors], out=offsets[1:])
            embedding = pa.ListArray.from_arrays(
                pa.array(offsets, type=pa.int32()),
                pa.array(np.concatenate(vectors), type=pa.float32()),
            )
            table = pa.Table.from_arrays(
                [
                    pa.array([row.record_id for row in rows], type=pa.string()),
                    pa.array([row.chunk_id for row in rows], type=pa.string()),
                    pa.array([row.chunk_index for row in rows], type=pa.int32()),
                    pa.array([row.text for row in rows], type=pa.string()),
                    pa.array([row.token_count for row in rows], type=pa.int32()),
                    embedding,
                ],
                schema=schema,
            )
            writer.write_table(table, row_group_size=row_group_size)
            written += len(rows)
    os.replace(tmp_path, path)
    logger.info(
        "parquet_exported", extra={"path": str(path), "rows": written, "model_key": model_key}
    )
    return written


def build_pipeline(
    config: PipelineConfig,
    *,
    client: AsyncOpenAI | None = None,
    with_embedder: bool = True,
    show_progress: bool = True,
    run_id: str | None = None,
) -> EmbeddingPipeline:
    """Wire up a pipeline from configuration.

    Args:
        config: Validated configuration.
        client: Optional injected ``AsyncOpenAI`` (or mock).
        with_embedder: Build the embedder (requires credentials unless ``client`` is given).
        show_progress: Render a progress bar.
        run_id: Optional explicit run id.

    Returns:
        A ready-to-run ``EmbeddingPipeline`` (close it, or use it as a context manager).
    """
    run_id = run_id or uuid.uuid4().hex[:12]
    store = CheckpointStore(config.storage.checkpoint_db)
    try:
        embedder = (
            AsyncEmbedder.from_config(config, run_id=run_id, client=client)
            if with_embedder
            else None
        )
    except BaseException:
        store.close()
        raise
    return EmbeddingPipeline(
        config, store=store, embedder=embedder, run_id=run_id, show_progress=show_progress
    )

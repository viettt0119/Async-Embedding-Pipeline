"""Tests for checkpoint persistence and crash/interrupt recovery of the full pipeline."""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pyarrow.parquet as pq
import pytest
from openai.types import CreateEmbeddingResponse

from src.checkpoint import CheckpointStore
from src.chunker import make_chunk_id
from src.embedder import DeadLetterQueue, FatalEmbeddingError
from src.pipeline import build_pipeline
from src.schemas import Chunk, EmbeddingResult, PipelineConfig

DIM = 4


class SimulatedCrash(BaseException):
    """Stands in for a hard process failure (not an ordinary, retryable Exception)."""


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def make_result(index: int, *, record: str = "r") -> EmbeddingResult:
    chunk = Chunk(
        chunk_id=make_chunk_id("b" * 64, index, f"text {index}"),
        record_id=f"{record}-{index}",
        chunk_index=index,
        text=f"text {index}",
        token_count=3,
    )
    return EmbeddingResult(chunk=chunk, embedding=np.arange(DIM, dtype=np.float32) + index)


def vector_for(text: str) -> list[float]:
    """Deterministic fake embedding so outputs can be verified against inputs."""
    return [float(len(text)), float(sum(map(ord, text)) % 997), 1.0, -1.0]


def response_for(inputs: list[str]) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse.model_validate(
        {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [
                {"object": "embedding", "index": i, "embedding": vector_for(t)}
                for i, t in enumerate(inputs)
            ],
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
        }
    )


class RecordingClient:
    """Fake AsyncOpenAI: records every input it embeds; can crash after N calls."""

    def __init__(self, crash_after: int | None = None, block_after: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.crash_after = crash_after
        self.block_after = block_after
        self.blocked = asyncio.Event()
        self.embeddings = MagicMock()
        self.embeddings.create = AsyncMock(side_effect=self._create)

    async def _create(self, *, input: list[str], **_: Any) -> CreateEmbeddingResponse:
        if self.crash_after is not None and len(self.calls) >= self.crash_after:
            raise SimulatedCrash("power cut")
        if self.block_after is not None and len(self.calls) >= self.block_after:
            self.blocked.set()
            await asyncio.Event().wait()  # hang until cancelled (simulates Ctrl+C mid-request)
        self.calls.append(list(input))
        return response_for(input)

    @property
    def embedded_texts(self) -> list[str]:
        return [text for call in self.calls for text in call]


@pytest.fixture
def config(tmp_path: Path) -> PipelineConfig:
    csv_path = tmp_path / "input.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "title", "body"])
        for i in range(40):
            body = " ".join(f"word{i}_{j}" for j in range(60 if i % 7 == 0 else 8))
            writer.writerow([f"doc-{i}", f"Title {i}", body])
        writer.writerow(["dup-1", "Title 1", " ".join(f"word1_{j}" for j in range(8))])  # duplicate
        writer.writerow(["empty", "", ""])

    return PipelineConfig.model_validate(
        {
            "data": {"input_csv": csv_path, "text_columns": ["title", "body"], "csv_chunksize": 7},
            "chunking": {"max_tokens": 48, "overlap_tokens": 8},
            "batching": {"max_items": 5, "max_tokens_per_batch": 400},
            "rate_limit": {
                "max_concurrency": 1,
                "requests_per_minute": None,
                "tokens_per_minute": None,
            },
            "retry": {"max_attempts": 2, "initial_wait_seconds": 0.001, "max_wait_seconds": 0.01},
            "storage": {
                "checkpoint_db": tmp_path / "state" / "checkpoint.db",
                "dead_letter_path": tmp_path / "state" / "dead_letter.jsonl",
                "output_parquet": tmp_path / "out" / "embeddings.parquet",
            },
            "runtime": {"pending_lookup_size": 6},
        }
    )


def expected_chunks(config: PipelineConfig) -> list[Chunk]:
    with build_pipeline(config, with_embedder=False, show_progress=False) as pipeline:
        return [chunk for batch in pipeline.iter_pending_batches() for chunk in batch.chunks]


async def run_pipeline(config: PipelineConfig, client: RecordingClient) -> Any:
    with build_pipeline(config, client=client, show_progress=False) as pipeline:  # type: ignore[arg-type]
        return await pipeline.run()


# --------------------------------------------------------------------------------------
# CheckpointStore unit tests
# --------------------------------------------------------------------------------------


class TestCheckpointStore:
    def test_save_and_reload_after_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "cp.db"
        results = [make_result(i) for i in range(5)]
        with CheckpointStore(db) as store:
            assert store.save_results(results, "m@native") == 5

        with CheckpointStore(db) as reopened:  # simulates a process restart
            assert reopened.completed_count("m@native") == 5
            ids = [r.chunk.chunk_id for r in results]
            assert reopened.pending_chunk_ids([*ids, "f" * 32], "m@native") == {"f" * 32}
            assert reopened.is_completed(ids[0], "m@native")

    def test_save_is_idempotent(self, tmp_path: Path) -> None:
        with CheckpointStore(tmp_path / "cp.db") as store:
            results = [make_result(i) for i in range(3)]
            assert store.save_results(results, "m") == 3
            assert store.save_results(results, "m") == 0  # replayed batch
            assert store.save_results([*results, make_result(3)], "m") == 1
            assert store.completed_count("m") == 4

    def test_models_are_isolated(self, tmp_path: Path) -> None:
        with CheckpointStore(tmp_path / "cp.db") as store:
            result = make_result(0)
            store.save_results([result], "small@native")
            assert store.pending_chunk_ids([result.chunk.chunk_id], "large@native") == {
                result.chunk.chunk_id
            }
            assert store.model_keys() == ["small@native"]

    def test_vectors_roundtrip_in_insertion_order(self, tmp_path: Path) -> None:
        with CheckpointStore(tmp_path / "cp.db") as store:
            results = [make_result(i) for i in range(25)]
            store.save_results(results[:10], "m")
            store.save_results(results[10:], "m")
            rows = [row for page in store.iter_results("m", batch_size=7) for row in page]
        assert [r.chunk_id for r in rows] == [r.chunk.chunk_id for r in results]
        for row, result in zip(rows, results, strict=True):
            np.testing.assert_array_equal(row.embedding, result.embedding)
            assert row.record_id == result.chunk.record_id

    def test_failed_write_rolls_back_whole_batch(self, tmp_path: Path) -> None:
        with CheckpointStore(tmp_path / "cp.db") as store:
            good = make_result(0)
            bad = EmbeddingResult(chunk=make_result(1).chunk, embedding=None)  # type: ignore[arg-type]
            with pytest.raises(AttributeError):
                store.save_results([good, bad], "m")
            assert store.completed_count("m") == 0

    def test_large_lookup_exceeds_sql_parameter_window(self, tmp_path: Path) -> None:
        with CheckpointStore(tmp_path / "cp.db") as store:
            results = [make_result(i) for i in range(2500)]
            store.save_results(results, "m")
            ids = [r.chunk.chunk_id for r in results]
            assert store.pending_chunk_ids(ids, "m") == set()


# --------------------------------------------------------------------------------------
# End-to-end recovery
# --------------------------------------------------------------------------------------


async def test_full_run_exports_every_chunk_once(config: PipelineConfig) -> None:
    chunks = expected_chunks(config)
    client = RecordingClient()

    report = await run_pipeline(config, client)

    assert report.chunks_embedded == len(chunks) == report.rows_exported
    assert report.duplicates_dropped == 1
    assert report.empty_rows_dropped == 1
    table = pq.read_table(config.storage.output_parquet)
    assert table.column_names == [
        "record_id",
        "chunk_id",
        "chunk_index",
        "text",
        "token_count",
        "embedding",
    ]
    assert table.num_rows == len(chunks)
    assert len(set(table.column("chunk_id").to_pylist())) == len(chunks)
    for text, vector in zip(
        table.column("text").to_pylist(), table.column("embedding").to_pylist(), strict=True
    ):
        assert vector == pytest.approx(vector_for(text))
    assert table.schema.metadata[b"model_key"] == b"text-embedding-3-small@native"


async def test_resume_after_crash_only_processes_remaining(config: PipelineConfig) -> None:
    chunks = expected_chunks(config)
    batch_size = config.batching.max_items

    # Run 1: hard crash on the 4th API call -> 3 batches were committed.
    crashing = RecordingClient(crash_after=3)
    with pytest.raises(SimulatedCrash):
        await run_pipeline(config, crashing)
    with CheckpointStore(config.storage.checkpoint_db) as store:
        committed = store.completed_count(config.embedding.model_key)
    assert committed == len(crashing.embedded_texts) == 3 * batch_size
    assert not config.storage.output_parquet.exists()

    # Run 2: fresh process state, same inputs -> only the remainder is sent to the API.
    resumed = RecordingClient()
    report = await run_pipeline(config, resumed)
    assert report.chunks_skipped == committed
    assert report.chunks_embedded == len(chunks) - committed
    assert set(resumed.embedded_texts).isdisjoint(crashing.embedded_texts)
    assert sorted(crashing.embedded_texts + resumed.embedded_texts) == sorted(
        c.text for c in chunks
    )

    table = pq.read_table(config.storage.output_parquet)
    assert table.num_rows == len(chunks)
    assert sorted(table.column("chunk_id").to_pylist()) == sorted(c.chunk_id for c in chunks)

    # Run 3: everything done -> idempotent, zero API calls, same output.
    idle = RecordingClient()
    report = await run_pipeline(config, idle)
    assert idle.calls == []
    assert report.chunks_skipped == len(chunks)
    assert report.rows_exported == len(chunks)


async def test_resume_after_cancellation_ctrl_c(config: PipelineConfig) -> None:
    chunks = expected_chunks(config)
    client = RecordingClient(block_after=2)

    task = asyncio.create_task(run_pipeline(config, client))
    await asyncio.wait_for(client.blocked.wait(), timeout=5)
    task.cancel()  # what asyncio.run does on SIGINT
    with pytest.raises(asyncio.CancelledError):
        await task

    with CheckpointStore(config.storage.checkpoint_db) as store:
        assert store.completed_count(config.embedding.model_key) == len(client.embedded_texts)

    resumed = RecordingClient()
    report = await run_pipeline(config, resumed)
    assert len(client.embedded_texts) + len(resumed.embedded_texts) == len(chunks)
    assert report.rows_exported == len(chunks)


async def test_dead_lettered_batches_are_retried_on_next_run(config: PipelineConfig) -> None:
    import openai

    from tests.test_embedder import api_error

    chunks = expected_chunks(config)
    poisoned = {chunks[0].text}

    class FlakyClient(RecordingClient):
        async def _create(self, *, input: list[str], **kw: Any) -> CreateEmbeddingResponse:
            if poisoned & set(input):
                raise api_error(openai.InternalServerError, 500)
            return await super()._create(input=input, **kw)

    report = await run_pipeline(config, FlakyClient())
    assert report.batches_failed == 1
    assert report.rows_exported == len(chunks) - config.batching.max_items
    assert DeadLetterQueue(config.storage.dead_letter_path).count() == 1

    poisoned.clear()  # upstream issue fixed
    retry_client = RecordingClient()
    report = await run_pipeline(config, retry_client)
    assert len(retry_client.embedded_texts) == config.batching.max_items
    assert report.rows_exported == len(chunks)


async def test_outage_trips_circuit_breaker_and_keeps_checkpoint(config: PipelineConfig) -> None:
    import openai

    from tests.test_embedder import _REQUEST

    chunks = expected_chunks(config)
    config = config.with_overrides({"retry.max_consecutive_failures": 2})

    class OutageClient(RecordingClient):
        async def _create(self, *, input: list[str], **kw: Any) -> CreateEmbeddingResponse:
            if len(self.calls) >= 2:  # network goes down after two good batches
                raise openai.APIConnectionError(request=_REQUEST)
            return await super()._create(input=input, **kw)

    client = OutageClient()
    with pytest.raises(FatalEmbeddingError, match="circuit breaker"):
        await run_pipeline(config, client)

    with CheckpointStore(config.storage.checkpoint_db) as store:
        assert store.completed_count(config.embedding.model_key) == len(client.embedded_texts)
    assert DeadLetterQueue(config.storage.dead_letter_path).count() == 2  # not the whole dataset

    report = await run_pipeline(config, RecordingClient())  # network back -> resume
    assert report.rows_exported == len(chunks)

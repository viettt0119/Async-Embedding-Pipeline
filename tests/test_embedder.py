"""Tests for the async embedder: retries on 429/5xx, DLQ, fatal errors, concurrency, limits."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx2
import numpy as np
import openai
import pytest
from openai.types import CreateEmbeddingResponse

from src.chunker import iter_batches, make_chunk_id
from src.embedder import (
    AsyncEmbedder,
    DeadLetterQueue,
    FatalEmbeddingError,
    RateLimiter,
    TokenBucket,
    is_retryable,
    retry_after_seconds,
)
from src.schemas import Batch, Chunk, RetrySettings

DIM = 8
_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/embeddings")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def api_error(
    cls: type[openai.APIStatusError],
    status: int,
    *,
    headers: dict[str, str] | None = None,
    code: str | None = None,
) -> openai.APIStatusError:
    body = {"error": {"message": f"HTTP {status}", "type": "test", "code": code}}
    response = httpx2.Response(status, request=_REQUEST, headers=headers or {}, json=body)
    return cls(f"HTTP {status}", response=response, body=body["error"])


def rate_limit_error(**kwargs: Any) -> openai.RateLimitError:
    return api_error(openai.RateLimitError, 429, **kwargs)  # type: ignore[return-value]


def embedding_response(inputs: list[str], *, shuffle: bool = False) -> CreateEmbeddingResponse:
    items = [
        {
            "object": "embedding",
            "index": i,
            "embedding": [float(len(text))] + [float(i)] * (DIM - 1),
        }
        for i, text in enumerate(inputs)
    ]
    if shuffle:
        items.reverse()
    return CreateEmbeddingResponse.model_validate(
        {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": items,
            "usage": {"prompt_tokens": 10 * len(inputs), "total_tokens": 10 * len(inputs)},
        }
    )


async def echo_create(*, input: list[str], **_: Any) -> CreateEmbeddingResponse:
    return embedding_response(input)


def make_batch(size: int = 3, prefix: str = "chunk") -> Batch:
    chunks = [
        Chunk(
            chunk_id=make_chunk_id("a" * 64, i, f"{prefix} {i}"),
            record_id=f"rec-{i}",
            chunk_index=0,
            text=f"{prefix} {i}",
            token_count=5,
        )
        for i in range(size)
    ]
    return next(iter_batches(chunks, max_items=size, max_tokens=10_000))


def mock_client(side_effect: Any) -> MagicMock:
    client = MagicMock()
    client.embeddings.create = AsyncMock(side_effect=side_effect)
    return client


def make_embedder(
    client: MagicMock,
    tmp_path: Path,
    *,
    max_attempts: int = 4,
    max_concurrency: int = 4,
    dimensions: int | None = None,
    respect_retry_after: bool = True,
    max_consecutive_failures: int | None = 10,
    rate_limiter: RateLimiter | None = None,
) -> AsyncEmbedder:
    return AsyncEmbedder(
        client,
        model="text-embedding-3-small",
        model_key="text-embedding-3-small@native",
        dimensions=dimensions,
        request_timeout=5.0,
        max_concurrency=max_concurrency,
        retry=RetrySettings(
            max_attempts=max_attempts,
            initial_wait_seconds=0.001,
            max_wait_seconds=0.01,
            respect_retry_after=respect_retry_after,
            max_consecutive_failures=max_consecutive_failures,
        ),
        rate_limiter=rate_limiter,
        dead_letter_queue=DeadLetterQueue(tmp_path / "dead_letter.jsonl"),
        run_id="test-run",
    )


# --------------------------------------------------------------------------------------
# Retry behaviour
# --------------------------------------------------------------------------------------


async def test_success_returns_vectors_in_request_order(tmp_path: Path) -> None:
    client = mock_client(lambda *, input, **_: embedding_response(input, shuffle=True))
    embedder = make_embedder(client, tmp_path)
    batch = make_batch(4)

    outcome = await embedder.embed_batch(batch)

    assert outcome.succeeded and outcome.attempts == 1
    assert [r.chunk.chunk_id for r in outcome.results] == [c.chunk_id for c in batch.chunks]
    for i, result in enumerate(outcome.results):
        assert result.embedding.dtype == np.float32
        assert result.embedding.shape == (DIM,)
        assert result.embedding[1] == i  # index-sorted despite the shuffled response
    kwargs = client.embeddings.create.await_args.kwargs
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["input"] == [c.text for c in batch.chunks]
    assert "dimensions" not in kwargs
    assert embedder.stats.tokens_used == 40


async def test_retries_rate_limit_429_then_succeeds(tmp_path: Path) -> None:
    client = mock_client(
        [rate_limit_error(), rate_limit_error(), embedding_response(["a", "b", "c"])]
    )
    embedder = make_embedder(client, tmp_path)

    outcome = await embedder.embed_batch(make_batch(3))

    assert outcome.succeeded
    assert outcome.attempts == 3
    assert client.embeddings.create.await_count == 3
    assert embedder.stats.retries == 2
    assert embedder.stats.requests == 3
    assert embedder.stats.failed_batches == 0
    assert not (tmp_path / "dead_letter.jsonl").exists()


@pytest.mark.parametrize(
    "error",
    [
        api_error(openai.InternalServerError, 500),
        api_error(openai.InternalServerError, 503),
        api_error(openai.APIStatusError, 502),
        openai.APITimeoutError(request=_REQUEST),
        openai.APIConnectionError(request=_REQUEST),
    ],
    ids=["500", "503", "502", "timeout", "connection"],
)
async def test_retries_transient_server_errors(tmp_path: Path, error: Exception) -> None:
    client = mock_client([error, embedding_response(["a", "b", "c"])])
    embedder = make_embedder(client, tmp_path)

    outcome = await embedder.embed_batch(make_batch(3))

    assert outcome.succeeded and outcome.attempts == 2
    assert embedder.stats.retries == 1


async def test_persistent_429_goes_to_dead_letter_queue_without_raising(tmp_path: Path) -> None:
    client = mock_client(rate_limit_error())  # every call fails
    embedder = make_embedder(client, tmp_path, max_attempts=3)
    batch = make_batch(2)

    outcome = await embedder.embed_batch(batch)

    assert not outcome.succeeded
    assert "RateLimitError" in (outcome.error or "")
    assert client.embeddings.create.await_count == 3
    assert embedder.stats.failed_batches == 1

    entries = embedder.dead_letter_queue.read_all()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.batch_id == batch.batch_id
    assert entry.status_code == 429
    assert entry.attempts == 3
    assert entry.run_id == "test-run"
    assert [c.chunk_id for c in entry.chunks] == [c.chunk_id for c in batch.chunks]
    assert [c.text for c in entry.chunks] == [c.text for c in batch.chunks]


async def test_non_retryable_400_dead_lettered_immediately(tmp_path: Path) -> None:
    client = mock_client(api_error(openai.BadRequestError, 400))
    embedder = make_embedder(client, tmp_path, max_attempts=5)

    outcome = await embedder.embed_batch(make_batch(1))

    assert not outcome.succeeded
    assert client.embeddings.create.await_count == 1
    assert embedder.stats.retries == 0
    assert embedder.dead_letter_queue.count() == 1


@pytest.mark.parametrize(
    "error",
    [
        api_error(openai.AuthenticationError, 401),
        api_error(openai.PermissionDeniedError, 403),
        rate_limit_error(code="insufficient_quota"),
    ],
    ids=["401", "403", "quota"],
)
async def test_fatal_errors_abort_without_retry(tmp_path: Path, error: Exception) -> None:
    client = mock_client(error)
    embedder = make_embedder(client, tmp_path)

    with pytest.raises(FatalEmbeddingError):
        await embedder.embed_batch(make_batch(1))
    assert client.embeddings.create.await_count == 1
    assert embedder.dead_letter_queue.count() == 0


async def test_malformed_response_is_dead_lettered(tmp_path: Path) -> None:
    client = mock_client(lambda *, input, **_: embedding_response(input[:-1]))
    embedder = make_embedder(client, tmp_path)

    outcome = await embedder.embed_batch(make_batch(3))

    assert not outcome.succeeded
    assert "EmbeddingResponseError" in (outcome.error or "")
    assert client.embeddings.create.await_count == 1  # not a transient error


async def test_dimensions_forwarded_and_validated(tmp_path: Path) -> None:
    client = mock_client(echo_create)
    embedder = make_embedder(client, tmp_path, dimensions=DIM)
    outcome = await embedder.embed_batch(make_batch(2))
    assert outcome.succeeded
    assert client.embeddings.create.await_args.kwargs["dimensions"] == DIM

    wrong = make_embedder(mock_client(echo_create), tmp_path, dimensions=DIM * 2)
    assert not (await wrong.embed_batch(make_batch(2))).succeeded


# --------------------------------------------------------------------------------------
# Retry-After handling
# --------------------------------------------------------------------------------------


def test_retry_after_parsing() -> None:
    assert retry_after_seconds(rate_limit_error(headers={"retry-after": "2"})) == 2.0
    assert retry_after_seconds(rate_limit_error(headers={"retry-after-ms": "250"})) == 0.25
    assert retry_after_seconds(rate_limit_error(headers={"retry-after": "Wed, 21 Oct"})) is None
    assert retry_after_seconds(rate_limit_error()) is None


async def test_retry_after_header_extends_backoff(tmp_path: Path) -> None:
    client = mock_client(
        [rate_limit_error(headers={"retry-after-ms": "5"}), embedding_response(["a"])]
    )
    embedder = make_embedder(client, tmp_path)
    sleeps: list[float] = []
    original = embedder._before_sleep

    def capture(batch: Batch):  # type: ignore[no-untyped-def]
        inner = original(batch)

        def hook(state):  # type: ignore[no-untyped-def]
            sleeps.append(state.upcoming_sleep)
            inner(state)

        return hook

    embedder._before_sleep = capture  # type: ignore[method-assign]
    assert (await embedder.embed_batch(make_batch(1))).succeeded
    assert sleeps and sleeps[0] >= 0.005


def test_retryable_classification() -> None:
    assert is_retryable(rate_limit_error())
    assert is_retryable(api_error(openai.APIStatusError, 408))
    assert is_retryable(api_error(openai.InternalServerError, 500))
    assert not is_retryable(api_error(openai.BadRequestError, 400))
    assert not is_retryable(api_error(openai.NotFoundError, 404))
    assert not is_retryable(rate_limit_error(code="insufficient_quota"))
    assert not is_retryable(ValueError("boom"))


# --------------------------------------------------------------------------------------
# Async batching / concurrency
# --------------------------------------------------------------------------------------


async def test_semaphore_bounds_in_flight_requests(tmp_path: Path) -> None:
    in_flight = 0
    peak = 0

    async def slow_create(*, input: list[str], **_: Any) -> CreateEmbeddingResponse:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return embedding_response(input)

    embedder = make_embedder(mock_client(slow_create), tmp_path, max_concurrency=3)
    batches = [make_batch(2, prefix=f"b{i}") for i in range(12)]

    outcomes = await asyncio.gather(*(embedder.embed_batch(b) for b in batches))

    assert all(o.succeeded for o in outcomes)
    assert peak == 3
    assert embedder.stats.succeeded_batches == 12


async def test_concurrent_mixed_failures_isolated(tmp_path: Path) -> None:
    """One permanently failing batch must not affect the others."""

    async def create(*, input: list[str], **_: Any) -> CreateEmbeddingResponse:
        if input[0].startswith("poison"):
            raise api_error(openai.InternalServerError, 500)
        await asyncio.sleep(0)
        return embedding_response(input)

    embedder = make_embedder(mock_client(create), tmp_path, max_attempts=2)
    batches = [make_batch(2, prefix=f"ok{i}") for i in range(5)] + [make_batch(2, prefix="poison")]

    outcomes = await asyncio.gather(*(embedder.embed_batch(b) for b in batches))

    assert sum(o.succeeded for o in outcomes) == 5
    assert embedder.stats.failed_batches == 1
    assert embedder.dead_letter_queue.count() == 1


# --------------------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------------------


async def test_token_bucket_blocks_until_refilled() -> None:
    now = [0.0]
    bucket = TokenBucket(per_minute=60, clock=lambda: now[0])  # 1 token/s, capacity 60
    await bucket.acquire(60)  # drain the burst capacity

    task = asyncio.create_task(bucket.acquire(2))
    await asyncio.sleep(0)
    assert not task.done()  # empty bucket -> waiting
    now[0] = 2.0  # two seconds later, two tokens have refilled
    await asyncio.wait_for(task, timeout=3)


async def test_token_bucket_clamps_oversized_requests() -> None:
    bucket = TokenBucket(per_minute=10)
    await asyncio.wait_for(bucket.acquire(1_000), timeout=1)  # would deadlock if not clamped


async def test_rate_limiter_disabled_buckets_are_noops() -> None:
    await asyncio.wait_for(RateLimiter(None, None).acquire(10**9), timeout=1)


# --------------------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------------------


async def test_circuit_breaker_trips_after_consecutive_failures(tmp_path: Path) -> None:
    client = mock_client(openai.APIConnectionError(request=_REQUEST))  # total outage
    embedder = make_embedder(client, tmp_path, max_attempts=2, max_consecutive_failures=3)

    for i in range(2):
        assert not (await embedder.embed_batch(make_batch(1, prefix=f"b{i}"))).succeeded
    with pytest.raises(FatalEmbeddingError, match="circuit breaker"):
        await embedder.embed_batch(make_batch(1, prefix="b2"))
    assert embedder.dead_letter_queue.count() == 3


async def test_circuit_breaker_resets_on_success(tmp_path: Path) -> None:
    async def create(*, input: list[str], **_: Any) -> CreateEmbeddingResponse:
        if input[0].startswith("bad"):
            raise api_error(openai.BadRequestError, 400)
        return embedding_response(input)

    embedder = make_embedder(mock_client(create), tmp_path, max_consecutive_failures=2)
    for i in range(5):  # bad, ok, bad, ok, ... never two failures in a row
        assert not (await embedder.embed_batch(make_batch(1, prefix=f"bad{i}"))).succeeded
        assert (await embedder.embed_batch(make_batch(1, prefix=f"ok{i}"))).succeeded
    assert embedder.stats.failed_batches == 5


async def test_circuit_breaker_can_be_disabled(tmp_path: Path) -> None:
    client = mock_client(api_error(openai.BadRequestError, 400))
    embedder = make_embedder(client, tmp_path, max_consecutive_failures=None)
    for i in range(25):
        assert not (await embedder.embed_batch(make_batch(1, prefix=f"b{i}"))).succeeded

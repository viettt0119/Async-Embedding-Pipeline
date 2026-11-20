"""Async OpenAI embeddings client with concurrency control, rate limiting, retries and a DLQ.

Fault-tolerance layers, from the inside out:

1. **Concurrency cap** - an ``asyncio.Semaphore`` bounds in-flight HTTP requests.
2. **Client-side rate limiting** - token buckets for requests/min and tokens/min smooth
   bursts *before* the server has to answer 429.
3. **Retries** - ``tenacity`` exponential backoff with full jitter on transient failures
   (429, timeouts, connection errors, 408/409, 5xx), honouring ``Retry-After`` headers.
   The SDK's own retries are disabled (``max_retries=0``) so there is exactly one policy.
4. **Dead-letter queue** - a batch that still fails (or fails non-retryably, e.g. 400) is
   appended to ``dead_letter.jsonl`` and the pipeline carries on. Those chunks are *not*
   checkpointed, so simply re-running the pipeline retries them.
5. **Fatal errors** - authentication/permission failures and exhausted quota would fail
   every batch identically, so they abort the run via ``FatalEmbeddingError``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import openai
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from src.schemas import (
    Batch,
    DeadLetterChunk,
    DeadLetterEntry,
    EmbeddingResult,
    PipelineConfig,
    RetrySettings,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
_FATAL_ERROR_CODES = frozenset({"insufficient_quota", "invalid_api_key"})


class FatalEmbeddingError(RuntimeError):
    """An error that would affect every batch (bad credentials, no quota): abort the run."""


class EmbeddingResponseError(RuntimeError):
    """The API answered successfully but the payload does not match the request."""


# --------------------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------------------


def is_fatal(exc: BaseException) -> bool:
    """Return whether ``exc`` should abort the whole run."""
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return True
    return (
        isinstance(exc, openai.APIStatusError) and getattr(exc, "code", None) in _FATAL_ERROR_CODES
    )


def is_retryable(exc: BaseException) -> bool:
    """Return whether ``exc`` is transient and worth retrying."""
    if is_fatal(exc):
        return False
    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES or exc.status_code >= 500
    return False


def retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a server-suggested delay from ``retry-after-ms`` / ``retry-after`` headers."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        if (millis := headers.get("retry-after-ms")) is not None:
            return max(float(millis) / 1000.0, 0.0)
        if (seconds := headers.get("retry-after")) is not None:
            return max(float(seconds), 0.0)
    except (TypeError, ValueError):  # HTTP-date form or garbage: fall back to backoff
        return None
    return None


class BackoffWithRetryAfter:
    """Full-jitter exponential backoff that never waits *less* than ``Retry-After``."""

    def __init__(self, settings: RetrySettings) -> None:
        """Create the wait strategy from the ``retry`` config section."""
        self._base = wait_random_exponential(
            multiplier=settings.initial_wait_seconds, max=settings.max_wait_seconds
        )
        self._respect_retry_after = settings.respect_retry_after
        self._max_wait = settings.max_wait_seconds

    def __call__(self, retry_state: RetryCallState) -> float:
        """Compute the next sleep for tenacity."""
        delay = float(self._base(retry_state))
        if self._respect_retry_after and retry_state.outcome is not None:
            hinted = retry_after_seconds(retry_state.outcome.exception())
            if hinted is not None:
                delay = max(delay, min(hinted, self._max_wait))
        return delay


# --------------------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------------------


class TokenBucket:
    """Async token bucket; capacity equals one minute's budget, refilled continuously."""

    def __init__(self, per_minute: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Create a bucket.

        Args:
            per_minute: Sustained budget per minute (also the burst capacity).
            clock: Monotonic clock (injectable for tests).
        """
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.capacity = float(per_minute)
        self._rate_per_second = per_minute / 60.0
        self._clock = clock
        self._tokens = self.capacity
        self._updated = clock()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0) -> None:
        """Wait until ``amount`` units are available, then consume them.

        Requests larger than the capacity are clamped so they cannot deadlock. Waiters are
        served FIFO because the lock is held while sleeping.
        """
        amount = min(float(amount), self.capacity)
        async with self._lock:
            while True:
                now = self._clock()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._updated) * self._rate_per_second
                )
                self._updated = now
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                await asyncio.sleep((amount - self._tokens) / self._rate_per_second)


class RateLimiter:
    """Combined requests-per-minute and tokens-per-minute limiter (either may be disabled)."""

    def __init__(self, requests_per_minute: float | None, tokens_per_minute: float | None) -> None:
        """Create the limiter; ``None`` disables the corresponding bucket."""
        self._requests = TokenBucket(requests_per_minute) if requests_per_minute else None
        self._tokens = TokenBucket(tokens_per_minute) if tokens_per_minute else None

    async def acquire(self, tokens: int) -> None:
        """Reserve one request carrying ``tokens`` input tokens."""
        if self._requests is not None:
            await self._requests.acquire(1)
        if self._tokens is not None:
            await self._tokens.acquire(tokens)


# --------------------------------------------------------------------------------------
# Dead-letter queue
# --------------------------------------------------------------------------------------


class DeadLetterQueue:
    """Append-only JSONL sink for batches that could not be embedded."""

    def __init__(self, path: Path | str) -> None:
        """Create the queue; the parent directory is created if needed."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def put(self, entry: DeadLetterEntry) -> None:
        """Durably append one entry (flushed and fsync'ed)."""
        line = entry.model_dump_json() + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def read_all(self) -> list[DeadLetterEntry]:
        """Load every entry (for inspection/replay tooling and tests)."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [DeadLetterEntry.model_validate_json(line) for line in handle if line.strip()]

    def count(self) -> int:
        """Number of entries currently in the queue file."""
        if not self.path.exists():
            return 0
        with self.path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())


# --------------------------------------------------------------------------------------
# Embedder
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class EmbedderStats:
    """Live counters (read by the progress bar and final report)."""

    requests: int = 0
    retries: int = 0
    succeeded_batches: int = 0
    failed_batches: int = 0
    tokens_used: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return counters as a plain dict."""
        return asdict(self)


@dataclass(slots=True)
class BatchOutcome:
    """Result of embedding one batch: either ``results`` or an ``error``."""

    batch: Batch
    attempts: int
    results: list[EmbeddingResult] = field(default_factory=list)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the batch was embedded."""
        return self.error is None


class AsyncEmbedder:
    """Embed batches concurrently with bounded parallelism, rate limiting and retries."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        model_key: str,
        dimensions: int | None,
        request_timeout: float,
        max_concurrency: int,
        retry: RetrySettings,
        rate_limiter: RateLimiter | None,
        dead_letter_queue: DeadLetterQueue,
        run_id: str,
    ) -> None:
        """Create an embedder.

        Args:
            client: ``AsyncOpenAI`` instance (inject a mock in tests). Its built-in retries
                should be disabled so that this class owns the retry policy.
            model: Embedding model name.
            model_key: Checkpoint namespace (recorded in dead-letter entries).
            dimensions: Optional output dimensionality (text-embedding-3-* only).
            request_timeout: Per-request timeout in seconds.
            max_concurrency: Maximum simultaneous in-flight requests.
            retry: Retry policy.
            rate_limiter: Client-side limiter, or ``None`` to disable.
            dead_letter_queue: Sink for unrecoverable batches.
            run_id: Identifier of the current pipeline run (for logs and the DLQ).
        """
        self._client = client
        self.model = model
        self.model_key = model_key
        self.dimensions = dimensions
        self.request_timeout = request_timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._retry_settings = retry
        self._wait = BackoffWithRetryAfter(retry)
        self._rate_limiter = rate_limiter
        self.dead_letter_queue = dead_letter_queue
        self.run_id = run_id
        self.stats = EmbedderStats()
        self._consecutive_failures = 0

    @classmethod
    def from_config(
        cls, config: PipelineConfig, *, run_id: str, client: AsyncOpenAI | None = None
    ) -> AsyncEmbedder:
        """Build an embedder (and, unless injected, an ``AsyncOpenAI`` client) from config.

        Raises:
            FatalEmbeddingError: If no client is injected and ``OPENAI_API_KEY`` is unset.
        """
        if client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise FatalEmbeddingError("OPENAI_API_KEY is not set (see .env.example)")
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=os.environ.get("OPENAI_BASE_URL") or None,
                max_retries=0,
                timeout=config.embedding.request_timeout_seconds,
            )
        limits = config.rate_limit
        return cls(
            client,
            model=config.embedding.model,
            model_key=config.embedding.model_key,
            dimensions=config.embedding.dimensions,
            request_timeout=config.embedding.request_timeout_seconds,
            max_concurrency=limits.max_concurrency,
            retry=config.retry,
            rate_limiter=RateLimiter(limits.requests_per_minute, limits.tokens_per_minute),
            dead_letter_queue=DeadLetterQueue(config.storage.dead_letter_path),
            run_id=run_id,
        )

    async def embed_batch(self, batch: Batch) -> BatchOutcome:
        """Embed one batch, retrying transient failures; never raises for per-batch errors.

        Args:
            batch: Chunks to embed in a single request.

        Returns:
            A ``BatchOutcome``; on failure the batch has already been dead-lettered.

        Raises:
            FatalEmbeddingError: On errors that would fail every batch (auth, quota), or
                when the circuit breaker trips after too many consecutive failed batches.
        """
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._retry_settings.max_attempts),
            wait=self._wait,
            retry=retry_if_exception(is_retryable),
            before_sleep=self._before_sleep(batch),
            reraise=True,
        )
        attempts = 0
        try:
            async for attempt in retrying:
                with attempt:
                    attempts = attempt.retry_state.attempt_number
                    vectors = await self._request(batch)
        except Exception as exc:
            if is_fatal(exc):
                raise FatalEmbeddingError(f"{type(exc).__name__}: {exc}") from exc
            await self._dead_letter(batch, exc, attempts)
            self._record_failure(exc)
            return BatchOutcome(
                batch=batch, attempts=attempts, error=f"{type(exc).__name__}: {exc}"
            )

        self._consecutive_failures = 0
        self.stats.succeeded_batches += 1
        results = [
            EmbeddingResult(chunk=chunk, embedding=vector)
            for chunk, vector in zip(batch.chunks, vectors, strict=True)
        ]
        return BatchOutcome(batch=batch, attempts=attempts, results=results)

    async def _request(self, batch: Batch) -> list[np.ndarray]:
        """Perform a single API call and validate the response shape."""
        options: dict[str, Any] = {"encoding_format": "float", "timeout": self.request_timeout}
        if self.dimensions is not None:
            options["dimensions"] = self.dimensions

        async with self._semaphore:
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire(batch.total_tokens)
            self.stats.requests += 1
            response = await self._client.embeddings.create(
                input=[chunk.text for chunk in batch.chunks], model=self.model, **options
            )

        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != batch.size or [item.index for item in data] != list(range(batch.size)):
            raise EmbeddingResponseError(
                f"expected {batch.size} embeddings with indices 0..{batch.size - 1}, "
                f"got {len(data)}"
            )
        vectors = [np.asarray(item.embedding, dtype=np.float32) for item in data]
        dims = {vector.shape[0] for vector in vectors}
        if len(dims) != 1 or (self.dimensions is not None and dims != {self.dimensions}):
            raise EmbeddingResponseError(f"inconsistent embedding dimensions: {sorted(dims)}")

        usage = getattr(response, "usage", None)
        self.stats.tokens_used += int(getattr(usage, "total_tokens", 0) or batch.total_tokens)
        return vectors

    def _before_sleep(self, batch: Batch) -> Callable[[RetryCallState], None]:
        def log_retry(state: RetryCallState) -> None:
            self.stats.retries += 1
            exc = state.outcome.exception() if state.outcome else None
            logger.warning(
                "batch_retry_scheduled",
                extra={
                    "batch_id": batch.batch_id,
                    "attempt": state.attempt_number,
                    "sleep_seconds": round(state.upcoming_sleep, 3),
                    "error_type": type(exc).__name__,
                    "status_code": getattr(exc, "status_code", None),
                },
            )

        return log_retry

    def _record_failure(self, exc: Exception) -> None:
        """Trip the circuit breaker after too many consecutive failed batches.

        Isolated bad batches (interleaved with successes) never trip it; a systemic outage
        does, so the run stops with its checkpoint intact instead of dead-lettering the
        entire remaining dataset.
        """
        self._consecutive_failures += 1
        limit = self._retry_settings.max_consecutive_failures
        if limit is not None and self._consecutive_failures >= limit:
            raise FatalEmbeddingError(
                f"circuit breaker open: {self._consecutive_failures} consecutive batches failed "
                f"(last error {type(exc).__name__}: {exc}); progress is checkpointed, fix the "
                "cause and re-run"
            )

    async def _dead_letter(self, batch: Batch, exc: Exception, attempts: int) -> None:
        self.stats.failed_batches += 1
        entry = DeadLetterEntry(
            run_id=self.run_id,
            batch_id=batch.batch_id,
            model_key=self.model_key,
            error_type=type(exc).__name__,
            error_message=str(exc)[:2000],
            status_code=getattr(exc, "status_code", None),
            attempts=max(attempts, 1),
            chunks=[
                DeadLetterChunk(
                    chunk_id=chunk.chunk_id,
                    record_id=chunk.record_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                )
                for chunk in batch.chunks
            ],
        )
        await self.dead_letter_queue.put(entry)
        logger.error(
            "batch_dead_lettered",
            extra={
                "batch_id": batch.batch_id,
                "chunks": batch.size,
                "attempts": entry.attempts,
                "error_type": entry.error_type,
                "status_code": entry.status_code,
                "dead_letter_path": str(self.dead_letter_queue.path),
            },
        )

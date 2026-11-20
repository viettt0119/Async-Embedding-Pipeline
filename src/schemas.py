"""Data contracts and validated configuration for the embedding pipeline.

Two families of models live here:

* **Configuration** (``PipelineConfig`` and its sections) - loaded from YAML, overridden by
  environment variables and CLI flags, and validated against the OpenAI Embeddings API's
  hard limits so that misconfiguration fails fast at start-up rather than mid-run.
* **Records** flowing through the pipeline (``CleanRecord`` → ``Chunk`` → ``Batch`` →
  ``EmbeddingResult``), plus the ``DeadLetterEntry`` written for unrecoverable batches.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Hard limits of the OpenAI Embeddings endpoint (per request / per input).
OPENAI_MAX_INPUT_TOKENS = 8191
OPENAI_MAX_INPUTS_PER_REQUEST = 2048
OPENAI_MAX_TOKENS_PER_REQUEST = 300_000

#: Environment variable -> dotted config path. Applied after YAML, before CLI flags.
ENV_OVERRIDES: dict[str, str] = {
    "EMBEDDING_MODEL": "embedding.model",
    "EMBEDDING_DIMENSIONS": "embedding.dimensions",
    "MAX_CONCURRENCY": "rate_limit.max_concurrency",
    "REQUESTS_PER_MINUTE": "rate_limit.requests_per_minute",
    "TOKENS_PER_MINUTE": "rate_limit.tokens_per_minute",
    "MAX_RETRY_ATTEMPTS": "retry.max_attempts",
    "BATCH_MAX_ITEMS": "batching.max_items",
}


class _FrozenModel(BaseModel):
    """Immutable, strict base model: unknown keys are rejected (catches YAML typos)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


class EmbeddingSettings(_FrozenModel):
    """Which embedding model to call and how."""

    model: str = Field(default="text-embedding-3-small", min_length=1)
    dimensions: int | None = Field(default=None, gt=0)
    request_timeout_seconds: float = Field(default=60.0, gt=0)

    @property
    def model_key(self) -> str:
        """Checkpoint namespace: vectors from different models/sizes must never mix."""
        return f"{self.model}@{self.dimensions or 'native'}"


class DataSettings(_FrozenModel):
    """Input CSV location and column mapping."""

    input_csv: Path = Path("data/raw/sample.csv")
    id_column: str = Field(default="id", min_length=1)
    text_columns: list[str] = Field(default_factory=lambda: ["text"], min_length=1)
    csv_chunksize: int = Field(default=10_000, gt=0)
    csv_encoding: str = "utf-8"
    min_text_chars: int = Field(default=1, ge=1)


class ChunkingSettings(_FrozenModel):
    """Token-based chunking parameters."""

    encoding: str = "cl100k_base"
    max_tokens: int = Field(default=512, gt=0, le=OPENAI_MAX_INPUT_TOKENS)
    overlap_tokens: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_window(self) -> Self:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be < max_tokens ({self.max_tokens})"
            )
        return self


class BatchingSettings(_FrozenModel):
    """Per-request batching limits (both are enforced simultaneously)."""

    max_items: int = Field(default=128, gt=0, le=OPENAI_MAX_INPUTS_PER_REQUEST)
    max_tokens_per_batch: int = Field(default=50_000, gt=0, le=OPENAI_MAX_TOKENS_PER_REQUEST)


class RateLimitSettings(_FrozenModel):
    """Client-side concurrency and throughput limits."""

    max_concurrency: int = Field(default=8, gt=0, le=512)
    requests_per_minute: float | None = Field(default=3000, gt=0)
    tokens_per_minute: float | None = Field(default=1_000_000, gt=0)


class RetrySettings(_FrozenModel):
    """Exponential-backoff-with-jitter retry policy."""

    max_attempts: int = Field(default=6, ge=1)
    initial_wait_seconds: float = Field(default=1.0, gt=0)
    max_wait_seconds: float = Field(default=60.0, gt=0)
    respect_retry_after: bool = True
    #: Circuit breaker: abort the run after this many *consecutive* dead-lettered batches
    #: (e.g. a full network/API outage) instead of dead-lettering the whole dataset.
    #: ``None`` disables the breaker.
    max_consecutive_failures: int | None = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _max_not_below_initial(self) -> Self:
        if self.max_wait_seconds < self.initial_wait_seconds:
            raise ValueError("max_wait_seconds must be >= initial_wait_seconds")
        return self


class StorageSettings(_FrozenModel):
    """Where state and outputs are persisted."""

    checkpoint_db: Path = Path("data/checkpoints/checkpoint.db")
    dead_letter_path: Path = Path("data/checkpoints/dead_letter.jsonl")
    output_parquet: Path = Path("data/processed/embeddings.parquet")
    parquet_row_group_size: int = Field(default=10_000, gt=0)


class RuntimeSettings(_FrozenModel):
    """Orchestrator tuning knobs."""

    pending_lookup_size: int = Field(default=1000, gt=0)
    queue_size_per_worker: int = Field(default=2, gt=0)


class PipelineConfig(_FrozenModel):
    """Root configuration object."""

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    batching: BatchingSettings = Field(default_factory=BatchingSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    @model_validator(mode="after")
    def _chunk_fits_in_batch(self) -> Self:
        if self.chunking.max_tokens > self.batching.max_tokens_per_batch:
            raise ValueError(
                "chunking.max_tokens must be <= batching.max_tokens_per_batch, otherwise a "
                "single chunk could never be sent"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path | str, env: Mapping[str, str] | None = None) -> Self:
        """Load configuration from YAML and apply environment overrides.

        Args:
            path: Path to the YAML file.
            env: Environment mapping to read overrides from (defaults to ``os.environ``).

        Returns:
            A validated configuration.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            pydantic.ValidationError: If the merged configuration is invalid.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top-level YAML must be a mapping")
        env = os.environ if env is None else env
        for variable, dotted in ENV_OVERRIDES.items():
            value = env.get(variable)
            if value not in (None, ""):
                _set_dotted(raw, dotted, value)
        return cls.model_validate(raw)

    def with_overrides(self, overrides: Mapping[str, Any]) -> Self:
        """Return a re-validated copy with dotted-path overrides applied.

        ``None`` values are ignored so that unset CLI flags don't clobber the config.

        Args:
            overrides: Mapping such as ``{"rate_limit.max_concurrency": 4}``.

        Returns:
            A new, validated configuration.
        """
        data = self.model_dump(mode="python")
        for dotted, value in overrides.items():
            if value is not None:
                _set_dotted(data, dotted, value)
        return type(self).model_validate(data)


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = target
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


class CleanRecord(_FrozenModel):
    """A normalised, de-duplicated input row."""

    record_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of `text`")


class Chunk(_FrozenModel):
    """A token-bounded slice of a record; the unit that gets embedded and checkpointed."""

    chunk_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    record_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)


class Batch(_FrozenModel):
    """A group of chunks sent in a single Embeddings API request."""

    batch_id: str = Field(min_length=1)
    chunks: tuple[Chunk, ...] = Field(min_length=1)

    @property
    def total_tokens(self) -> int:
        """Sum of the chunks' token counts (used for rate limiting)."""
        return sum(chunk.token_count for chunk in self.chunks)

    @property
    def size(self) -> int:
        """Number of chunks in the batch."""
        return len(self.chunks)


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """A chunk paired with its embedding vector.

    A slotted dataclass rather than a Pydantic model: this sits on the hot path and holds a
    ``float32`` ndarray, which Pydantic would otherwise re-validate element by element.
    """

    chunk: Chunk
    embedding: np.ndarray


class DeadLetterChunk(_FrozenModel):
    """Enough information to replay a single chunk."""

    chunk_id: str
    record_id: str
    chunk_index: int
    text: str


class DeadLetterEntry(_FrozenModel):
    """One line of ``dead_letter.jsonl``: a batch that exhausted its retries."""

    run_id: str
    batch_id: str
    model_key: str
    error_type: str
    error_message: str
    status_code: int | None = None
    attempts: int = Field(ge=1)
    failed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    chunks: list[DeadLetterChunk]

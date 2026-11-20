# Async Embedding Pipeline

> 🇻🇳 Vietnamese version: [README_vi.md](README_vi.md)

Production-grade, **resumable** pipeline that turns arbitrarily large CSV files into OpenAI
embeddings stored as Apache Parquet:

```
CSV (streamed) ─▶ clean + SHA-256 dedup ─▶ token chunking (tiktoken) ─▶ skip already-done
   ─▶ token-budgeted batches ─▶ async OpenAI calls (semaphore + rate limit + retry)
   ─▶ SQLite checkpoint (atomic per batch) ─▶ Parquet export
```

Design goals: bounded memory, no duplicate API spend, crash-safe resume, and failures that
degrade gracefully instead of killing a multi-hour run.

---

## Quick start

```bash
uv sync                                            # create .venv from uv.lock
cp .env.example .env                               # then set OPENAI_API_KEY
uv run python data/generate_sample_csv.py --rows 100000   # dirty sample data -> data/raw/sample.csv

uv run embed-pipeline run --dry-run                # volume/token estimate, no API calls
uv run embed-pipeline run                          # full run (Ctrl+C any time; re-run to resume)
uv run embed-pipeline stats                        # checkpoint + dead-letter counts
uv run embed-pipeline export                       # re-export Parquet from the checkpoint
```

Run the tests and the linter:

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

### CLI reference

```
embed-pipeline <command> [--config PATH] [--log-level LEVEL] [--log-format text|json] [options]

run     --input CSV --output PARQUET --checkpoint DB --model NAME --dimensions N
        --max-concurrency N --batch-size N --max-tokens N --overlap-tokens N
        --csv-chunksize N --dry-run --no-export --no-progress
export  --output PARQUET --checkpoint DB
stats   --checkpoint DB
```

`--config`, `--log-level` and `--log-format` can be given before or after the subcommand.

| Exit code | Meaning |
|---|---|
| `0` | success |
| `1` | configuration / input error (bad YAML, missing column, missing file) |
| `2` | finished, but some batches were dead-lettered: fix the cause and re-run |
| `3` | fatal: invalid key, permission denied, quota exhausted, or circuit breaker tripped |
| `130` | interrupted (Ctrl+C / `docker stop`): progress is saved, re-run to resume |

### Docker

```bash
docker build -t async-embedding-pipeline .

# Optional: generate sample data into ./data
docker run --rm -v "$PWD/data:/app/data" --entrypoint python \
  async-embedding-pipeline data/generate_sample_csv.py --rows 100000

docker run --rm -v "$PWD/data:/app/data" async-embedding-pipeline run --dry-run
docker run --rm --env-file .env -v "$PWD/data:/app/data" async-embedding-pipeline   # full run
docker run --rm -v "$PWD/data:/app/data" async-embedding-pipeline stats
```

- **Mount `data/` as a volume**, because the checkpoint lives there. Restarting the container
  with the same volume resumes where it stopped.
- **`docker stop` is graceful.** The image sets `STOPSIGNAL SIGINT`. Python running as PID 1
  ignores the default SIGTERM, which would mean a SIGKILL after 10 s. With SIGINT the run is
  cancelled cleanly: committed batches are kept and the container exits with code `130`.
- **Reproducible and self-contained.** Dependencies come from `uv.lock` (`--frozen`), the
  `cl100k_base` encoding is baked in so it isn't downloaded at runtime, the process runs as a
  non-root user, and logs are JSON (`LOG_FORMAT=json`).
- **WSL2 without Docker Desktop integration.** Call the Windows binary directly, e.g.
  `/mnt/c/.../DockerDesktop/resources/bin/docker.exe`, or enable integration in Docker
  Desktop → Settings → Resources → WSL integration.

**Verified scenario (20,000 dirty rows, 21,198 chunks, fake OpenAI endpoint injecting about
13% 429/500 errors):**
- `docker stop` mid-run: exit `130` within 1 s, with 7,936 chunks checkpointed.
- Restart: skipped those 7,936, absorbed 17 retries, exported 21,198 rows.
- Third run: 0 API calls.
- Full API outage: the circuit breaker aborted after 76 s (exit `3`) with 10 DLQ entries,
  not 166.

---

## Project layout

```
configs/pipeline_config.yaml   Operational parameters (model, batching, limits, retry policy)
data/generate_sample_csv.py    Deterministic generator of large, deliberately dirty CSVs
data/raw/                      Input CSVs
data/processed/                embeddings.parquet
data/checkpoints/              checkpoint.db (SQLite) + dead_letter.jsonl
src/schemas.py                 Pydantic config + record contracts, validated against API limits
src/cleaner.py                 Streaming CSV reader, normalisation, content-hash dedup
src/chunker.py                 Token windows with overlap; item- and token-bounded batching
src/checkpoint.py              SQLite store: completed chunks + their vectors
src/embedder.py                AsyncOpenAI + Semaphore + token buckets + tenacity + DLQ + breaker
src/pipeline.py                Producer/consumer orchestrator, tqdm progress, Parquet export
src/cli.py                     CLI, structured logging, exit codes
tests/                         pytest + pytest-asyncio (mocked OpenAI, crash/resume tests)
```

The package is importable as `src` (e.g. `from src.pipeline import build_pipeline`), as the
target layout requires.

---

## Architecture

### Data flow and concurrency

```
┌──────────────── worker thread ────────────────┐         ┌───────────── event loop ─────────────┐
│ pandas.read_csv(chunksize)                     │         │  N consumer tasks (N = concurrency)  │
│   → normalise → SHA-256 dedup                  │ bounded │    embed_batch()                     │
│   → TokenChunker (512 tok, 50 overlap)         │ asyncio │      Semaphore → RateLimiter → API   │
│   → checkpoint lookup (skip completed ids)     │ ──────▶ │      tenacity retry / DLQ / breaker  │
│   → BatchBuilder (≤ items, ≤ tokens)           │  Queue  │    save_results() (SQLite, thread)   │
└────────────────────────────────────────────────┘         │    tqdm update                        │
                                                           └───────────────────────────────────────┘
```

- **Bounded memory.** Rows are read in pandas chunks, and the producer is a generator advanced
  one batch at a time. The queue between the stages is bounded
  (`max_concurrency × queue_size_per_worker`), so reading never gets far ahead of embedding.
- **The event loop never blocks.** CSV parsing and tokenisation are CPU-bound, so the producer
  runs in a worker thread via `asyncio.to_thread`, as do SQLite commits. The event loop only
  drives HTTP.
- **Progress.** `tqdm.asyncio` shows chunks/s and ETA, plus tokens used, retries, failed
  batches and skipped (already-done) chunks. The total grows as the stream is discovered and is
  exact once reading finishes. Logs are routed through tqdm so the bar isn't torn apart. The
  bar is shown only when stderr is a TTY, so Docker and CI get clean JSON logs.

### Cleaning (`cleaner.py`)

1. The configured `text_columns` are concatenated, and missing columns fail fast.
2. Each value goes through Unicode NFKC normalisation. C0/C1 control characters, zero-width
   characters, BOMs, line separators and `U+FFFD` are removed, and whitespace is collapsed.
3. Rows that are empty or shorter than `min_text_chars` are dropped. A missing id is replaced
   with `row-<n>`.
4. Rows are de-duplicated by **SHA-256 of the normalised text**: the first occurrence wins, so
   ids stay stable between runs. The index holds 32-byte digests, about 1 GB per 10M unique rows.

### Chunking and batching (`chunker.py`)

- `cl100k_base` gives exact token counts. A text of up to `max_tokens` is kept as is. Longer
  texts are split into windows of `max_tokens` that start every `max_tokens − overlap_tokens`
  tokens, so consecutive chunks share `overlap_tokens` of context.
- **Multi-byte safety.** Window edges are moved by up to 3 tokens onto UTF-8 character
  boundaries, so Vietnamese, CJK and emoji never decode into `U+FFFD`. Each chunk is re-counted
  after decoding so the `max_tokens` limit is guaranteed.
- Batches are bounded by **both** `max_items` and `max_tokens_per_batch`. Config validation
  enforces the API's hard limits: 8,191 tokens per input, 2,048 inputs and 300k tokens per
  request.

---

## Fault tolerance

| Failure | Mechanism | Effect |
|---|---|---|
| Crash, `kill`, Ctrl+C, `docker stop`, network loss | Each successful batch is committed **with its vectors** in one SQLite transaction (WAL). On start-up, chunks whose ids are already in the checkpoint are skipped. | At most the in-flight batches are lost and re-sent. Completed work is never paid for twice. |
| Replayed batch (crash between API response and commit) | `INSERT OR IGNORE` on `(chunk_id, model_key)` | Idempotent: no duplicates. |
| 429 / timeout / connection error / 408 / 409 / 5xx | `tenacity`: exponential backoff with **full jitter**, capped at `max_wait_seconds`, never shorter than `Retry-After` / `retry-after-ms` | Transient errors are absorbed. Jitter stops workers from retrying in lockstep. |
| Bursts that would cause 429s | `asyncio.Semaphore(max_concurrency)` plus client-side **token buckets** for requests/min and tokens/min | Rate limits are rarely hit in the first place. |
| Batch still failing after all retries, or non-retryable (400, malformed response) | Appended (fsync'ed) to **`dead_letter.jsonl`** with error, status, attempts, run id and the chunk texts | The run continues and exits with code `2`. Failed chunks are not checkpointed, so **re-running retries them automatically**. |
| Systemic outage (API or network down for a long time) | **Circuit breaker**: `retry.max_consecutive_failures` (default 10) dead-lettered batches *in a row* abort the run. A success resets the counter. | Fails in about a minute (exit `3`) instead of grinding through and dead-lettering the whole dataset. Re-run once the API is back. |
| Invalid key / permission denied / `insufficient_quota` | `FatalEmbeddingError` | The run aborts immediately (exit `3`) instead of dead-lettering every batch. |
| Crash during export | Parquet is written to a temp file, then `os.replace`d | Readers never see a partial file. |

The SDK's built-in retries are disabled (`max_retries=0`), so tenacity is the only retry policy.

### Idempotency keys

- `content_hash = sha256(normalised_text)`
- `chunk_id = sha256(content_hash : chunk_index : sha256(chunk_text))[:32]`
- The checkpoint is namespaced by `model_key = "<model>@<dimensions|native>"`.

Same input and same parameters always give the same ids. Changing `max_tokens` or
`overlap_tokens` gives new ids, and changing the model gives a new namespace, so a stale
checkpoint is never applied to the wrong output.

### Durability trade-off

The checkpoint uses `synchronous=NORMAL` in WAL mode. Committed batches survive a process
crash. After a sudden power loss, the last few commits may be rolled back, and those chunks
are simply embedded again on the next run.

---

## Configuration

`configs/pipeline_config.yaml` is the source of truth. Values are applied in this order, each
overriding the previous one:

1. YAML
2. Environment variables / `.env`: `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`,
   `MAX_CONCURRENCY`, `REQUESTS_PER_MINUTE`, `TOKENS_PER_MINUTE`, `MAX_RETRY_ATTEMPTS`,
   `BATCH_MAX_ITEMS`
3. CLI flags

Credentials (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`) are read only from the environment.
They are never part of the config object and never logged. Unknown YAML keys are rejected, so
a typo fails at start-up instead of being silently ignored.

**Tuning:**
- Set `requests_per_minute` and `tokens_per_minute` slightly below your organisation's tier
  limits.
- Raise `max_concurrency` until the token bucket, not latency, is the bottleneck.
- Run `--dry-run` first: `estimated_tokens` × the model's price per token gives the cost.

---

## Output

`data/processed/embeddings.parquet` (zstd-compressed, written in row groups):

| column | type |
|---|---|
| `record_id` | string |
| `chunk_id` | string |
| `chunk_index` | int32 |
| `text` | string |
| `token_count` | int32 |
| `embedding` | list<float32> |

The file metadata includes `model_key`.

```python
import pyarrow.parquet as pq
import numpy as np

table = pq.read_table("data/processed/embeddings.parquet")
matrix = np.stack(table.column("embedding").to_numpy(zero_copy_only=False))
```

---

## Testing

| File | Covers |
|---|---|
| `test_cleaner_chunker.py` | Normalisation, NaN/empty dropping, cross-chunk SHA-256 dedup, missing columns; token limits, overlap correctness, multi-byte safety, deterministic ids; item- and token-bounded batching; config validation against API limits |
| `test_checkpoint.py` | Persistence across reopen, idempotent inserts, model isolation, rollback on failed writes; **end-to-end crash recovery** (hard crash mid-run, then resume sends only the remainder and a third run makes zero API calls), **Ctrl+C cancellation** recovery, dead-lettered batches retried on the next run, **outage → circuit breaker → resume** |
| `test_embedder.py` | `AsyncMock` OpenAI client: retries on 429/500/502/503/timeout/connection errors, DLQ after exhausting retries, no retry on 400, abort on 401/403/quota, `Retry-After` handling, response-order validation, semaphore concurrency bound, failure isolation, circuit breaker (trips, resets, can be disabled), token-bucket behaviour |
| `test_cli.py` | Global options before and after the subcommand (the Docker `CMD` form), exit codes for dry run, missing API key and invalid config |

The mocks use real `openai` exception classes, built from real HTTP responses, and real
`CreateEmbeddingResponse` models, so a change to the SDK's interface makes the tests fail.

---

## Limitations and extensions

- The de-duplication index is held in memory. For billions of rows, swap in a disk-backed
  set (e.g. a SQLite table) or a Bloom filter.
- When records are duplicates, only the first occurrence's `record_id` appears in the output.
  If downstream consumers need every id, store an alias table.
- The checkpoint is a single SQLite file, which assumes one pipeline process per checkpoint.
  For horizontal scale, shard the input and use one checkpoint per shard.
- The checkpoint also stores chunk text, so it is roughly as large as the Parquet output. This
  is the price of an export that can always be rebuilt from the checkpoint.

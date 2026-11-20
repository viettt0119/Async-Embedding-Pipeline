"""Production-grade async data & embedding pipeline.

Modules:
    schemas: Pydantic data contracts and validated configuration.
    cleaner: Streaming CSV reader, text normalisation and content-hash de-duplication.
    chunker: Token-aware chunking (tiktoken) and token-budgeted batching.
    checkpoint: SQLite-backed progress store enabling idempotent resume.
    embedder: Async OpenAI client with concurrency control, rate limiting, retries and DLQ.
    pipeline: Orchestrator wiring everything together, plus Parquet export.
    cli: Command-line interface.
"""

__version__ = "0.1.0"

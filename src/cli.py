"""Command-line interface for the embedding pipeline.

Examples:
    uv run embed-pipeline run
    uv run embed-pipeline run --input data/raw/big.csv --max-concurrency 16 --log-format json
    uv run embed-pipeline run --dry-run
    uv run embed-pipeline export --output data/processed/embeddings.parquet
    uv run embed-pipeline stats

Exit codes:
    0  success
    1  configuration / input error
    2  finished, but some batches were dead-lettered (re-run to retry them)
    3  fatal API error (authentication, permissions, quota)
    130 interrupted (Ctrl+C); progress is checkpointed, re-run to resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError
from tqdm.contrib.logging import logging_redirect_tqdm

from src.checkpoint import CheckpointStore
from src.cleaner import MissingColumnError
from src.embedder import DeadLetterQueue, FatalEmbeddingError
from src.pipeline import PipelineReport, build_pipeline
from src.schemas import PipelineConfig

logger = logging.getLogger("src.cli")

EXIT_OK = 0
EXIT_CONFIG_ERROR = 1
EXIT_PARTIAL_FAILURE = 2
EXIT_FATAL_API_ERROR = 3
EXIT_INTERRUPTED = 130

_STANDARD_LOG_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
    | {"message", "asctime", "taskName"}
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line; ``extra={...}`` fields become top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record`` as JSON."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(
            {key: value for key, value in record.__dict__.items() if key not in _STANDARD_LOG_ATTRS}
        )
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class KeyValueFormatter(logging.Formatter):
    """Human-readable ``time level logger event key=value ...`` lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` with its structured extras appended."""
        base = super().format(record)
        extras = " ".join(
            f"{key}={value}"
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_ATTRS
        )
        return f"{base} {extras}".rstrip()


def configure_logging(level: str, fmt: str) -> None:
    """Configure root logging to stderr.

    Args:
        level: Log level name (``DEBUG``, ``INFO``...).
        fmt: ``json`` for structured logs, ``text`` for key=value lines.
    """
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            KeyValueFormatter("%(asctime)s %(levelname)-7s %(name)s %(message)s", "%H:%M:%S")
        )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpx2", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _add_global_options(parser: argparse.ArgumentParser, *, suppress_defaults: bool) -> None:
    """Register options accepted both before and after the subcommand.

    Subparsers use ``SUPPRESS`` defaults so that an option given *before* the subcommand
    (``embed-pipeline --config x run``) is not overwritten by the subparser's default, while
    ``embed-pipeline run --config x`` works too.
    """

    def default(value: Any) -> Any:
        return argparse.SUPPRESS if suppress_defaults else value

    parser.add_argument(
        "--config",
        type=Path,
        default=default(Path("configs/pipeline_config.yaml")),
        help="YAML config path (default: configs/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--log-level", default=default(os.environ.get("LOG_LEVEL", "INFO")), help="log level"
    )
    parser.add_argument(
        "--log-format",
        choices=("text", "json"),
        default=default(os.environ.get("LOG_FORMAT", "text")),
        help="log output format",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="embed-pipeline",
        description="Resumable async CSV → OpenAI embeddings → Parquet pipeline.",
    )
    _add_global_options(parser, suppress_defaults=False)
    common = argparse.ArgumentParser(add_help=False)
    _add_global_options(common, suppress_defaults=True)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common], help="clean, chunk, embed and export")
    run.add_argument("--input", type=Path, help="override data.input_csv")
    run.add_argument("--output", type=Path, help="override storage.output_parquet")
    run.add_argument("--checkpoint", type=Path, help="override storage.checkpoint_db")
    run.add_argument("--model", help="override embedding.model")
    run.add_argument("--dimensions", type=int, help="override embedding.dimensions")
    run.add_argument("--max-concurrency", type=int, help="override rate_limit.max_concurrency")
    run.add_argument("--batch-size", type=int, help="override batching.max_items")
    run.add_argument("--max-tokens", type=int, help="override chunking.max_tokens")
    run.add_argument("--overlap-tokens", type=int, help="override chunking.overlap_tokens")
    run.add_argument("--csv-chunksize", type=int, help="override data.csv_chunksize")
    run.add_argument("--dry-run", action="store_true", help="estimate volume without API calls")
    run.add_argument("--no-export", action="store_true", help="skip the Parquet export step")
    run.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    run.set_defaults(handler=_cmd_run)

    export = sub.add_parser(
        "export", parents=[common], help="re-export Parquet from the checkpoint store"
    )
    export.add_argument("--output", type=Path, help="override storage.output_parquet")
    export.add_argument("--checkpoint", type=Path, help="override storage.checkpoint_db")
    export.set_defaults(handler=_cmd_export)

    stats = sub.add_parser("stats", parents=[common], help="show checkpoint and dead-letter counts")
    stats.add_argument("--checkpoint", type=Path, help="override storage.checkpoint_db")
    stats.set_defaults(handler=_cmd_stats)
    return parser


def load_config(args: argparse.Namespace) -> PipelineConfig:
    """Load YAML (+ env overrides) and apply CLI flag overrides."""
    config = PipelineConfig.from_yaml(args.config)
    overrides = {
        "data.input_csv": getattr(args, "input", None),
        "storage.output_parquet": getattr(args, "output", None),
        "storage.checkpoint_db": getattr(args, "checkpoint", None),
        "embedding.model": getattr(args, "model", None),
        "embedding.dimensions": getattr(args, "dimensions", None),
        "rate_limit.max_concurrency": getattr(args, "max_concurrency", None),
        "batching.max_items": getattr(args, "batch_size", None),
        "chunking.max_tokens": getattr(args, "max_tokens", None),
        "chunking.overlap_tokens": getattr(args, "overlap_tokens", None),
        "data.csv_chunksize": getattr(args, "csv_chunksize", None),
    }
    return config.with_overrides(overrides)


def _print_report(report: PipelineReport | dict[str, Any]) -> None:
    payload = report.as_dict() if isinstance(report, PipelineReport) else report
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _cmd_run(args: argparse.Namespace, config: PipelineConfig) -> int:
    show_progress = not args.no_progress and sys.stderr.isatty()

    if args.dry_run:
        with build_pipeline(config, with_embedder=False, show_progress=False) as pipeline:
            report = pipeline.plan()
        _print_report(report)
        return EXIT_OK

    async def _run() -> PipelineReport:
        with build_pipeline(config, show_progress=show_progress) as pipeline:
            return await pipeline.run(export=not args.no_export)

    with logging_redirect_tqdm():
        report = asyncio.run(_run())
    _print_report(report)
    if report.batches_failed:
        logger.warning(
            "run_finished_with_dead_letters",
            extra={
                "batches_failed": report.batches_failed,
                "dead_letter_path": str(config.storage.dead_letter_path),
                "hint": "fix the cause and re-run; failed chunks are retried automatically",
            },
        )
        return EXIT_PARTIAL_FAILURE
    return EXIT_OK


def _cmd_export(args: argparse.Namespace, config: PipelineConfig) -> int:
    with build_pipeline(config, with_embedder=False, show_progress=False) as pipeline:
        rows = pipeline.export(args.output)
    _print_report({"output_path": str(args.output or config.storage.output_parquet), "rows": rows})
    return EXIT_OK


def _cmd_stats(args: argparse.Namespace, config: PipelineConfig) -> int:
    with CheckpointStore(config.storage.checkpoint_db) as store:
        per_model = {key: store.completed_count(key) for key in store.model_keys()}
    _print_report(
        {
            "checkpoint_db": str(config.storage.checkpoint_db),
            "completed_chunks_by_model": per_model,
            "current_model_key": config.embedding.model_key,
            "dead_letter_path": str(config.storage.dead_letter_path),
            "dead_letter_entries": DeadLetterQueue(config.storage.dead_letter_path).count(),
        }
    )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    load_dotenv()
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.log_format)

    try:
        config = load_config(args)
        return int(args.handler(args, config))
    except (ValidationError, FileNotFoundError, MissingColumnError, ValueError) as exc:
        logger.error("configuration_or_input_error", extra={"error": str(exc)})
        return EXIT_CONFIG_ERROR
    except FatalEmbeddingError as exc:
        logger.error("fatal_api_error", extra={"error": str(exc)})
        return EXIT_FATAL_API_ERROR
    except KeyboardInterrupt:
        logger.warning(
            "interrupted",
            extra={"hint": "progress is checkpointed; re-run the same command to resume"},
        )
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())

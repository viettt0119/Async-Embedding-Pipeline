"""Generate a large, deliberately *dirty* CSV for exercising the embedding pipeline.

The output is fully deterministic for a given ``--seed`` (reproducibility), and it is
written row-by-row so that multi-million-row files can be produced in constant memory.

Injected data-quality problems (each controlled by a rate flag):
    * missing bodies / titles (NaN after parsing),
    * leading/trailing/internal whitespace noise,
    * control and zero-width characters,
    * exact duplicate texts under different ids (exercises SHA-256 de-duplication),
    * very long bodies (forces multi-chunk splitting with overlap),
    * multi-byte Vietnamese text (exercises token-boundary-safe decoding),
    * occasionally missing ids (the cleaner synthesises ``row-<n>`` ids).

Example:
    uv run python data/generate_sample_csv.py --rows 200000 --output data/raw/sample.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

_WORDS: tuple[str, ...] = (
    "data",
    "pipeline",
    "embedding",
    "vector",
    "token",
    "batch",
    "async",
    "retry",
    "latency",
    "throughput",
    "checkpoint",
    "resume",
    "schema",
    "parquet",
    "cluster",
    "semantic",
    "search",
    "retrieval",
    "model",
    "inference",
    "gradient",
    "storage",
    "stream",
    "queue",
    "backoff",
    "jitter",
    "idempotent",
    "partition",
    "shard",
    "replica",
    "cache",
    "index",
    "query",
    "context",
    "window",
    "overlap",
    "chunk",
    "customer",
    "invoice",
    "shipment",
    "warehouse",
    "ticket",
    "support",
    "refund",
    "product",
    "review",
    "rating",
    "quality",
    "delivery",
    "payment",
    "account",
)
_VIETNAMESE: tuple[str, ...] = (
    "Dữ liệu được xử lý theo lô và lưu trữ an toàn.",
    "Hệ thống tự động thử lại khi gặp lỗi giới hạn tốc độ.",
    "Khách hàng đánh giá sản phẩm rất cao về chất lượng.",
    "Đơn hàng đã được giao thành công đến kho trung tâm.",
    "Mô hình nhúng chuyển văn bản thành vector ngữ nghĩa.",
)
_CATEGORIES: tuple[str, ...] = ("support", "sales", "logistics", "product", "finance")
_NOISE: tuple[str, ...] = ("\x07", "\x0b", "\x1b", "\u200b", "\ufeff", "\x0c", "\u2028")


def _sentence(rng: random.Random, min_words: int = 6, max_words: int = 18) -> str:
    words = rng.choices(_WORDS, k=rng.randint(min_words, max_words))
    return " ".join(words).capitalize() + "."


def _paragraph(rng: random.Random, sentences: int) -> str:
    parts: list[str] = []
    for _ in range(sentences):
        if rng.random() < 0.15:
            parts.append(rng.choice(_VIETNAMESE))
        else:
            parts.append(_sentence(rng))
    return " ".join(parts)


def _dirty(rng: random.Random, text: str) -> str:
    """Sprinkle whitespace and control-character noise into ``text``."""
    if rng.random() < 0.5:
        text = f"{' ' * rng.randint(1, 4)}{text}{'  ' * rng.randint(1, 3)}"
    if rng.random() < 0.5:
        words = text.split(" ")
        position = rng.randrange(len(words))
        words[position] = words[position] + rng.choice(_NOISE)
        text = " ".join(words)
    if rng.random() < 0.3:
        text = text.replace(". ", ".\t\t  ", 1)
    return text


def generate(
    output: Path,
    rows: int,
    seed: int,
    duplicate_rate: float,
    null_rate: float,
    long_rate: float,
    dirty_rate: float,
) -> None:
    """Write ``rows`` synthetic records to ``output``.

    Args:
        output: Destination CSV path (parent directories are created).
        rows: Number of data rows to emit (excluding the header).
        seed: RNG seed; identical arguments always yield an identical file.
        duplicate_rate: Probability that a row re-uses an earlier title+body verbatim.
        null_rate: Probability that a row's body (and separately title) is empty.
        long_rate: Probability that a body is long enough to need several chunks.
        dirty_rate: Probability that whitespace/control-character noise is injected.
    """
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    epoch = datetime(2024, 1, 1, tzinfo=UTC)
    recent: list[tuple[str, str]] = []
    started = time.perf_counter()

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["id", "title", "body", "category", "created_at"])
        for index in range(rows):
            record_id = "" if rng.random() < 0.001 else f"doc-{index:09d}"

            if recent and rng.random() < duplicate_rate:
                # Exact re-post of an earlier document under a new id.
                title, body = rng.choice(recent)
            else:
                title = "" if rng.random() < null_rate else _sentence(rng, 3, 8)
                if rng.random() < null_rate:
                    body = ""
                else:
                    long_body = rng.random() < long_rate
                    sentences = rng.randint(60, 220) if long_body else rng.randint(1, 8)
                    body = _paragraph(rng, sentences)
                    if rng.random() < dirty_rate:
                        body = _dirty(rng, body)
                recent.append((title, body))
                if len(recent) > 1000:
                    recent.pop(0)

            created_at = epoch + timedelta(seconds=rng.randint(0, 3600 * 24 * 365))
            writer.writerow(
                [record_id, title, body, rng.choice(_CATEGORIES), created_at.isoformat()]
            )

            if index and index % 100_000 == 0:
                print(f"  ... {index:,} rows written", file=sys.stderr)

    elapsed = time.perf_counter() - started
    size_mb = output.stat().st_size / 1_048_576
    print(
        f"Wrote {rows:,} rows ({size_mb:,.1f} MiB) to {output} in {elapsed:.1f}s", file=sys.stderr
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=100_000, help="number of rows (default: 100k)")
    parser.add_argument("--output", type=Path, default=Path("data/raw/sample.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate-rate", type=float, default=0.05)
    parser.add_argument("--null-rate", type=float, default=0.02)
    parser.add_argument("--long-rate", type=float, default=0.03)
    parser.add_argument("--dirty-rate", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    if args.rows <= 0:
        raise SystemExit("--rows must be positive")
    generate(
        output=args.output,
        rows=args.rows,
        seed=args.seed,
        duplicate_rate=args.duplicate_rate,
        null_rate=args.null_rate,
        long_rate=args.long_rate,
        dirty_rate=args.dirty_rate,
    )


if __name__ == "__main__":
    main()

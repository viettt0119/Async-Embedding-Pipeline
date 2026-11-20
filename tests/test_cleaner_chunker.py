"""Tests for CSV cleaning/de-duplication and token-bounded chunking/batching."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.chunker import BatchBuilder, TokenChunker, iter_batches, make_chunk_id
from src.cleaner import CsvCleaner, MissingColumnError, content_hash, normalize_text
from src.schemas import Chunk, CleanRecord, PipelineConfig


def write_csv(path: Path, rows: list[dict[str, str]], header: list[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_record(text: str, record_id: str = "r1") -> CleanRecord:
    return CleanRecord(record_id=record_id, text=text, content_hash=content_hash(text))


def make_chunk(index: int, tokens: int) -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id("0" * 64, index, f"t{index}"),
        record_id="r",
        chunk_index=index,
        text=f"t{index}",
        token_count=tokens,
    )


@pytest.fixture(scope="module")
def chunker() -> TokenChunker:
    return TokenChunker(encoding_name="cl100k_base", max_tokens=32, overlap_tokens=8)


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------


class TestNormalizeText:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  hello   world \t\n", "hello world"),
            ("bell\x07char\x1bhere", "bell char here"),
            ("zero\u200bwidth\ufeff", "zero width"),
            ("ﬁ ligature", "fi ligature"),  # NFKC
            ("line\u2028sep", "line sep"),
        ],
    )
    def test_cleans(self, raw: str, expected: str) -> None:
        assert normalize_text(raw) == expected

    @pytest.mark.parametrize("raw", [None, float("nan"), "", "   ", "\x00\x07\t"])
    def test_empty_becomes_none(self, raw: object) -> None:
        assert normalize_text(raw) is None


class TestCsvCleaner:
    def test_drops_nan_empty_and_duplicates_across_chunks(self, tmp_path: Path) -> None:
        rows = [
            {"id": "a", "title": "Hello", "body": "  first   body "},
            {"id": "b", "title": "", "body": ""},  # empty -> dropped
            {"id": "c", "title": "Hello", "body": "first body"},  # dup after normalisation
            {"id": "d", "title": "Other", "body": "text\x07 with noise"},
            {"id": "", "title": "No", "body": "id here"},  # synthesised id
            {"id": "e", "title": "hello", "body": "first body"},  # case differs -> unique
        ]
        path = write_csv(tmp_path / "in.csv", rows, ["id", "title", "body"])
        cleaner = CsvCleaner(path, id_column="id", text_columns=["title", "body"], chunksize=2)

        records = list(cleaner.iter_records())

        assert [r.record_id for r in records] == ["a", "d", "row-4", "e"]
        assert records[0].text == "Hello first body"
        assert records[1].text == "Other text with noise"
        assert records[0].content_hash == content_hash("Hello first body")
        assert cleaner.stats.rows_read == 6
        assert cleaner.stats.dropped_empty == 1
        assert cleaner.stats.dropped_duplicates == 1
        assert cleaner.stats.synthesized_ids == 1
        assert cleaner.stats.emitted == 4

    def test_min_text_chars(self, tmp_path: Path) -> None:
        path = write_csv(
            tmp_path / "in.csv",
            [{"id": "1", "text": "ok"}, {"id": "2", "text": "long enough"}],
            ["id", "text"],
        )
        cleaner = CsvCleaner(path, id_column="id", text_columns=["text"], min_text_chars=5)
        assert [r.record_id for r in cleaner.iter_records()] == ["2"]
        assert cleaner.stats.dropped_too_short == 1

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path / "in.csv", [{"id": "1", "text": "x"}], ["id", "text"])
        cleaner = CsvCleaner(path, id_column="id", text_columns=["body"])
        with pytest.raises(MissingColumnError, match="body"):
            list(cleaner.iter_records())

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        cleaner = CsvCleaner(tmp_path / "nope.csv", id_column="id", text_columns=["text"])
        with pytest.raises(FileNotFoundError):
            list(cleaner.iter_records())


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------


class TestTokenChunker:
    def test_short_text_is_single_unchanged_chunk(self, chunker: TokenChunker) -> None:
        chunks = chunker.chunk_record(make_record("A short sentence."))
        assert len(chunks) == 1
        assert chunks[0].text == "A short sentence."
        assert chunks[0].token_count == chunker.count_tokens("A short sentence.")

    def test_every_chunk_respects_max_tokens(self, chunker: TokenChunker) -> None:
        text = " ".join(f"word{i} alpha beta gamma." for i in range(400))
        chunks = chunker.chunk_record(make_record(text))
        assert len(chunks) > 5
        for chunk in chunks:
            assert chunker.count_tokens(chunk.text) == chunk.token_count
            assert chunk.token_count <= chunker.max_tokens

    def test_overlap_between_consecutive_windows(self) -> None:
        chunker = TokenChunker(max_tokens=20, overlap_tokens=5)
        # Single-token words with leading spaces -> clean token boundaries.
        text = " ".join(["the"] * 3 + [f"w{i}" for i in range(80)])
        tokens = chunker.encode(text)
        pieces = chunker.split_text(text)
        assert len(pieces) >= 4
        step = chunker.max_tokens - chunker.overlap_tokens
        # Windows start every `step` tokens, so window i+1 re-covers the last 5 tokens of i.
        for i in range(len(pieces) - 1):
            window = tokens[i * step : i * step + chunker.max_tokens]
            next_window = tokens[(i + 1) * step : (i + 1) * step + chunker.max_tokens]
            assert window[-chunker.overlap_tokens :] == next_window[: chunker.overlap_tokens]
            tail = chunker._encoding.decode(window[-chunker.overlap_tokens :]).strip()
            assert pieces[i][0].endswith(tail)
            assert pieces[i + 1][0].startswith(tail)

    def test_last_chunk_reaches_end_of_text(self, chunker: TokenChunker) -> None:
        text = " ".join(f"token{i}" for i in range(300)) + " THE-END"
        pieces = chunker.split_text(text)
        assert pieces[-1][0].endswith("THE-END")
        assert pieces[0][0].startswith("token0")

    def test_multibyte_text_never_produces_replacement_chars(self) -> None:
        chunker = TokenChunker(max_tokens=17, overlap_tokens=4)
        text = "Dữ liệu được xử lý theo lô và lưu trữ an toàn. 🚀 数据管道 " * 40
        chunks = chunker.chunk_record(make_record(text))
        assert len(chunks) > 10
        for chunk in chunks:
            assert "\ufffd" not in chunk.text
            assert chunk.token_count <= 17

    def test_chunk_ids_are_deterministic_and_unique(self, chunker: TokenChunker) -> None:
        record = make_record(" ".join(f"item{i}" for i in range(200)))
        first = chunker.chunk_record(record)
        second = chunker.chunk_record(record)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert len({c.chunk_id for c in first}) == len(first)
        assert [c.chunk_index for c in first] == list(range(len(first)))

    def test_chunk_ids_change_with_chunking_params(self) -> None:
        record = make_record(" ".join(f"item{i}" for i in range(200)))
        a = TokenChunker(max_tokens=32, overlap_tokens=8).chunk_record(record)
        b = TokenChunker(max_tokens=40, overlap_tokens=8).chunk_record(record)
        assert {c.chunk_id for c in a}.isdisjoint({c.chunk_id for c in b})

    def test_special_token_text_is_treated_as_plain_text(self, chunker: TokenChunker) -> None:
        assert chunker.count_tokens("<|endoftext|>") > 1

    @pytest.mark.parametrize(("max_tokens", "overlap"), [(10, 10), (10, 12), (0, 0), (10, -1)])
    def test_invalid_params(self, max_tokens: int, overlap: int) -> None:
        with pytest.raises(ValueError):
            TokenChunker(max_tokens=max_tokens, overlap_tokens=overlap)


# --------------------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------------------


class TestBatching:
    def test_respects_max_items(self) -> None:
        batches = list(
            iter_batches([make_chunk(i, 1) for i in range(10)], max_items=4, max_tokens=999)
        )
        assert [b.size for b in batches] == [4, 4, 2]

    def test_respects_max_tokens(self) -> None:
        chunks = [make_chunk(i, t) for i, t in enumerate([40, 40, 30, 50, 10, 100])]
        batches = list(iter_batches(chunks, max_items=100, max_tokens=100))
        for batch in batches:
            assert batch.total_tokens <= 100
        assert sum(b.size for b in batches) == len(chunks)
        assert [c.chunk_index for b in batches for c in b.chunks] == list(range(len(chunks)))

    def test_exact_token_packing(self) -> None:
        chunks = [make_chunk(i, t) for i, t in enumerate([40, 40, 30, 50, 10, 100])]
        batches = list(iter_batches(chunks, max_items=100, max_tokens=100))
        assert [[c.token_count for c in b.chunks] for b in batches] == [
            [40, 40],
            [30, 50, 10],
            [100],
        ]

    def test_oversized_chunk_rejected(self) -> None:
        builder = BatchBuilder(max_items=5, max_tokens=10)
        with pytest.raises(ValueError, match="exceeding"):
            builder.add(make_chunk(0, 11))

    def test_batch_ids_deterministic(self) -> None:
        chunks = [make_chunk(i, 1) for i in range(6)]
        a = [b.batch_id for b in iter_batches(chunks, max_items=3, max_tokens=99)]
        b = [b.batch_id for b in iter_batches(chunks, max_items=3, max_tokens=99)]
        assert a == b and len(set(a)) == 2

    def test_flush_empty_returns_none(self) -> None:
        assert BatchBuilder(max_items=1, max_tokens=1).flush() is None


# --------------------------------------------------------------------------------------
# Config validation (guards the API limits the chunker/batcher rely on)
# --------------------------------------------------------------------------------------


class TestConfig:
    def test_repo_config_is_valid(self) -> None:
        config = PipelineConfig.from_yaml("configs/pipeline_config.yaml", env={})
        assert config.chunking.max_tokens == 512
        assert config.chunking.overlap_tokens == 50

    def test_env_and_cli_overrides(self) -> None:
        config = PipelineConfig.from_yaml(
            "configs/pipeline_config.yaml", env={"MAX_CONCURRENCY": "3", "EMBEDDING_MODEL": "m"}
        )
        assert config.rate_limit.max_concurrency == 3
        assert config.embedding.model_key == "m@native"
        updated = config.with_overrides(
            {"rate_limit.max_concurrency": 5, "chunking.max_tokens": None}
        )
        assert updated.rate_limit.max_concurrency == 5
        assert updated.chunking.max_tokens == 512

    @pytest.mark.parametrize(
        "override",
        [
            {"chunking.overlap_tokens": 512},
            {"chunking.max_tokens": 9000},
            {"batching.max_items": 5000},
            {"batching.max_tokens_per_batch": 100},  # smaller than max_tokens
            {"retry.max_wait_seconds": 0.1},
        ],
    )
    def test_invalid_configs_rejected(self, override: dict[str, int]) -> None:
        config = PipelineConfig.from_yaml("configs/pipeline_config.yaml", env={})
        with pytest.raises(ValueError):
            config.with_overrides(override)

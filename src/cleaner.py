"""Streaming CSV ingestion, text normalisation and content-hash de-duplication.

The CSV is read with ``pandas.read_csv(chunksize=...)`` so memory stays bounded by the chunk
size regardless of file size. Each row is:

1. projected onto ``id_column`` + ``text_columns`` (missing columns fail fast),
2. normalised (Unicode NFKC, control/zero-width characters removed, whitespace collapsed),
3. dropped if empty/too short,
4. de-duplicated by SHA-256 of the normalised text so identical texts are embedded once.

Memory note: the de-dup index keeps one 32-byte digest per *unique* text (~100 bytes with
set overhead), i.e. roughly 1 GB per 10M unique rows.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.schemas import CleanRecord, DataSettings

logger = logging.getLogger(__name__)

# C0/C1 control chars except \t \n \r (handled as whitespace), plus zero-width / BOM /
# line-separator code points and the U+FFFD replacement character from bad decodes.
_CONTROL_CHARS = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u2028\u2029\u2060\ufeff\ufffd]"
)
_WHITESPACE = re.compile(r"\s+")


class MissingColumnError(ValueError):
    """Raised when the CSV lacks a configured id/text column."""


@dataclass(slots=True)
class CleaningStats:
    """Counters describing what the cleaner did."""

    rows_read: int = 0
    dropped_empty: int = 0
    dropped_too_short: int = 0
    dropped_duplicates: int = 0
    synthesized_ids: int = 0
    emitted: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return the counters as a plain dict (for logging/reporting)."""
        return asdict(self)


def normalize_text(value: Any) -> str | None:
    """Normalise a raw cell value into clean single-spaced text.

    Args:
        value: Raw cell value (``str``, ``NaN``, ``None`` or anything ``str()``-able).

    Returns:
        The cleaned text, or ``None`` if nothing meaningful remains.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # array-likes; treat as text
        pass
    text = unicodedata.normalize("NFKC", str(value))
    text = _CONTROL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text or None


def content_hash(text: str) -> str:
    """Return the hex SHA-256 digest of ``text`` (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CsvCleaner:
    """Stream clean, unique records out of an arbitrarily large CSV file."""

    def __init__(
        self,
        path: Path | str,
        *,
        id_column: str,
        text_columns: Sequence[str],
        chunksize: int = 10_000,
        encoding: str = "utf-8",
        min_text_chars: int = 1,
    ) -> None:
        """Create a cleaner.

        Args:
            path: CSV file to read.
            id_column: Column holding the record id.
            text_columns: Columns whose cleaned values are joined (space-separated) to form
                the text to embed.
            chunksize: Rows per pandas chunk; bounds peak memory.
            encoding: File encoding. Undecodable bytes are replaced, then stripped.
            min_text_chars: Records whose cleaned text is shorter than this are dropped.
        """
        if not text_columns:
            raise ValueError("text_columns must not be empty")
        self.path = Path(path)
        self.id_column = id_column
        self.text_columns = list(text_columns)
        self.chunksize = chunksize
        self.encoding = encoding
        self.min_text_chars = min_text_chars
        self.stats = CleaningStats()
        self._seen_digests: set[bytes] = set()

    @classmethod
    def from_settings(cls, settings: DataSettings) -> CsvCleaner:
        """Build a cleaner from the ``data`` config section."""
        return cls(
            settings.input_csv,
            id_column=settings.id_column,
            text_columns=settings.text_columns,
            chunksize=settings.csv_chunksize,
            encoding=settings.csv_encoding,
            min_text_chars=settings.min_text_chars,
        )

    def iter_records(self) -> Iterator[CleanRecord]:
        """Yield clean, unique records in file order.

        Yields:
            ``CleanRecord`` objects; the first occurrence of a duplicated text wins, which
            keeps record ids stable across re-runs over the same file.

        Raises:
            FileNotFoundError: If the CSV does not exist.
            MissingColumnError: If a configured column is absent from the header.
        """
        if not self.path.is_file():
            raise FileNotFoundError(f"Input CSV not found: {self.path}")

        wanted = [self.id_column, *self.text_columns]
        logger.info(
            "csv_read_started",
            extra={"path": str(self.path), "chunksize": self.chunksize, "columns": wanted},
        )
        row_offset = 0
        with pd.read_csv(
            self.path,
            chunksize=self.chunksize,
            dtype=str,
            encoding=self.encoding,
            encoding_errors="replace",
            on_bad_lines="warn",
            keep_default_na=True,
        ) as reader:
            for frame in reader:
                missing = [column for column in wanted if column not in frame.columns]
                if missing:
                    raise MissingColumnError(
                        f"{self.path}: missing column(s) {missing}; found {list(frame.columns)}"
                    )
                yield from self._clean_frame(frame, row_offset)
                row_offset += len(frame)
                logger.debug("csv_chunk_processed", extra=self.stats.as_dict())

        logger.info("csv_read_finished", extra=self.stats.as_dict())

    def _clean_frame(self, frame: pd.DataFrame, row_offset: int) -> Iterator[CleanRecord]:
        ids = frame[self.id_column].tolist()
        columns = [frame[column].tolist() for column in self.text_columns]
        stats = self.stats

        for position, raw_id in enumerate(ids):
            stats.rows_read += 1
            parts = [part for column in columns if (part := normalize_text(column[position]))]
            if not parts:
                stats.dropped_empty += 1
                continue
            text = " ".join(parts)
            if len(text) < self.min_text_chars:
                stats.dropped_too_short += 1
                continue

            digest = hashlib.sha256(text.encode("utf-8")).digest()
            if digest in self._seen_digests:
                stats.dropped_duplicates += 1
                continue
            self._seen_digests.add(digest)

            record_id = normalize_text(raw_id)
            if record_id is None:
                record_id = f"row-{row_offset + position}"
                stats.synthesized_ids += 1

            stats.emitted += 1
            yield CleanRecord(record_id=record_id, text=text, content_hash=digest.hex())

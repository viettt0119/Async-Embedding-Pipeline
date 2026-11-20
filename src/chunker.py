"""Token-aware chunking (tiktoken) and token-budgeted batching.

* ``TokenChunker`` slides a window of ``max_tokens`` over a record's tokens with
  ``overlap_tokens`` of shared context between consecutive chunks. Window edges are nudged
  onto UTF-8 character boundaries so multi-byte scripts (e.g. Vietnamese) never decode into
  ``U+FFFD`` garbage, and every emitted chunk is re-counted to guarantee the token bound.
* ``BatchBuilder`` packs chunks into requests bounded by *both* item count and total tokens.

Chunk ids are deterministic: ``sha256(record_content_hash : chunk_index : sha256(chunk_text))``
truncated to 128 bits. Identical input + identical chunking parameters ⇒ identical ids, which
is what makes checkpoint-based resume idempotent. Changing ``max_tokens``/``overlap_tokens``
changes chunk texts and therefore ids, so stale checkpoints are never mis-applied.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator

import tiktoken

from src.schemas import Batch, BatchingSettings, Chunk, ChunkingSettings, CleanRecord

# A UTF-8 code point spans at most 4 bytes; a split character can therefore straddle at most
# 3 extra tokens on either side of a window edge.
_MAX_BOUNDARY_SHIFT = 4


def make_chunk_id(record_hash: str, chunk_index: int, text: str) -> str:
    """Deterministic 128-bit chunk id derived from content and position.

    Args:
        record_hash: SHA-256 hex digest of the parent record's text.
        chunk_index: Position of the chunk within the record.
        text: The chunk text.

    Returns:
        A 32-character lowercase hex string.
    """
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return hashlib.sha256(f"{record_hash}:{chunk_index}:{text_hash}".encode()).hexdigest()[:32]


class TokenChunker:
    """Split text into overlapping windows bounded by a token budget."""

    def __init__(
        self, *, encoding_name: str = "cl100k_base", max_tokens: int = 512, overlap_tokens: int = 50
    ) -> None:
        """Create a chunker.

        Args:
            encoding_name: tiktoken encoding (``cl100k_base`` matches OpenAI embedding models).
            max_tokens: Hard upper bound on tokens per chunk.
            overlap_tokens: Tokens shared between consecutive chunks; must be < ``max_tokens``.

        Raises:
            ValueError: On non-positive ``max_tokens`` or an overlap that is too large.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must satisfy 0 <= overlap_tokens < max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._encoding = tiktoken.get_encoding(encoding_name)

    @classmethod
    def from_settings(cls, settings: ChunkingSettings) -> TokenChunker:
        """Build a chunker from the ``chunking`` config section."""
        return cls(
            encoding_name=settings.encoding,
            max_tokens=settings.max_tokens,
            overlap_tokens=settings.overlap_tokens,
        )

    def encode(self, text: str) -> list[int]:
        """Tokenise ``text``; special-token strings are treated as plain text."""
        return self._encoding.encode(text, disallowed_special=())

    def count_tokens(self, text: str) -> int:
        """Return the exact token count of ``text``."""
        return len(self.encode(text))

    def split_text(self, text: str) -> list[tuple[str, int]]:
        """Split ``text`` into ``(chunk_text, token_count)`` windows.

        Args:
            text: Already-normalised text.

        Returns:
            Windows in order. A text that fits in one window is returned unchanged.
        """
        tokens = self.encode(text)
        total = len(tokens)
        if total == 0:
            return []
        if total <= self.max_tokens:
            return [(text, total)]

        step = self.max_tokens - self.overlap_tokens
        pieces: list[tuple[str, int]] = []
        start = 0
        while True:
            end = min(start + self.max_tokens, total)
            begin, stop = self._align(tokens, start, end)
            piece, count = self._decode_within_budget(tokens, begin, stop)
            if piece:
                pieces.append((piece, count))
            if end >= total:
                break
            # Never jump past the (possibly trimmed) end of the previous window: no gaps.
            start = max(min(start + step, stop), start + 1)
        return pieces

    def chunk_record(self, record: CleanRecord) -> list[Chunk]:
        """Split a record into ``Chunk`` objects with deterministic ids."""
        return [
            Chunk(
                chunk_id=make_chunk_id(record.content_hash, index, piece),
                record_id=record.record_id,
                chunk_index=index,
                text=piece,
                token_count=count,
            )
            for index, (piece, count) in enumerate(self.split_text(record.text))
        ]

    def _align(self, tokens: list[int], start: int, end: int) -> tuple[int, int]:
        """Shift window edges (by < 4 tokens) so the window decodes to valid UTF-8."""
        start_shifts = range(_MAX_BOUNDARY_SHIFT) if start > 0 else range(1)
        end_shifts = range(_MAX_BOUNDARY_SHIFT) if end < len(tokens) else range(1)
        for start_shift in start_shifts:
            for end_shift in end_shifts:
                begin, stop = start + start_shift, end - end_shift
                if stop <= begin:
                    break
                try:
                    self._encoding.decode_bytes(tokens[begin:stop]).decode("utf-8")
                except UnicodeDecodeError:
                    continue
                return begin, stop
        return start, end

    def _decode_within_budget(self, tokens: list[int], begin: int, stop: int) -> tuple[str, int]:
        """Decode a window, trimming it if re-tokenisation would exceed ``max_tokens``.

        BPE is context-sensitive, so a decoded slice can (rarely) re-encode to a different
        length. Embedding requests are metered on the *text*, so the text is re-counted.
        """
        while stop > begin:
            piece = self._encoding.decode(tokens[begin:stop]).strip()
            count = self.count_tokens(piece)
            if count <= self.max_tokens:
                return piece, count
            stop -= 1
        return "", 0


def make_batch_id(chunks: Iterable[Chunk]) -> str:
    """Deterministic batch id: hash of the member chunk ids."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("ascii"))
    return digest.hexdigest()[:16]


class BatchBuilder:
    """Incrementally pack chunks into batches bounded by item count and total tokens."""

    def __init__(self, *, max_items: int, max_tokens: int) -> None:
        """Create a builder.

        Args:
            max_items: Maximum chunks per batch.
            max_tokens: Maximum summed ``token_count`` per batch.
        """
        if max_items <= 0 or max_tokens <= 0:
            raise ValueError("max_items and max_tokens must be positive")
        self.max_items = max_items
        self.max_tokens = max_tokens
        self._pending: list[Chunk] = []
        self._pending_tokens = 0

    @classmethod
    def from_settings(cls, settings: BatchingSettings) -> BatchBuilder:
        """Build from the ``batching`` config section."""
        return cls(max_items=settings.max_items, max_tokens=settings.max_tokens_per_batch)

    def add(self, chunk: Chunk) -> Batch | None:
        """Add a chunk; return the previous batch if this chunk would overflow it.

        Raises:
            ValueError: If a single chunk exceeds ``max_tokens`` on its own.
        """
        if chunk.token_count > self.max_tokens:
            raise ValueError(
                f"chunk {chunk.chunk_id} has {chunk.token_count} tokens, "
                f"exceeding the per-batch limit of {self.max_tokens}"
            )
        completed: Batch | None = None
        if self._pending and (
            len(self._pending) >= self.max_items
            or self._pending_tokens + chunk.token_count > self.max_tokens
        ):
            completed = self.flush()
        self._pending.append(chunk)
        self._pending_tokens += chunk.token_count
        return completed

    def flush(self) -> Batch | None:
        """Emit whatever is pending (or ``None`` when empty)."""
        if not self._pending:
            return None
        batch = Batch(batch_id=make_batch_id(self._pending), chunks=tuple(self._pending))
        self._pending = []
        self._pending_tokens = 0
        return batch


def iter_batches(chunks: Iterable[Chunk], *, max_items: int, max_tokens: int) -> Iterator[Batch]:
    """Pack an iterable of chunks into batches (convenience wrapper around ``BatchBuilder``)."""
    builder = BatchBuilder(max_items=max_items, max_tokens=max_tokens)
    for chunk in chunks:
        if (batch := builder.add(chunk)) is not None:
            yield batch
    if (batch := builder.flush()) is not None:
        yield batch

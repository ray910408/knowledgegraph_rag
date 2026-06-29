from __future__ import annotations

from ..contracts import ProblemChunk
from .search_text import is_non_empty_chunk_text


class GenericFallbackChunker:
    def __init__(self, *, window_tokens: int = 200, overlap_tokens: int = 50) -> None:
        self._window_tokens = max(window_tokens, overlap_tokens + 1)
        self._overlap_tokens = overlap_tokens

    def chunk(self, text: str, *, input_id: str) -> tuple[ProblemChunk, ...]:
        normalized = text.strip()
        if not is_non_empty_chunk_text(normalized):
            return ()

        tokens = normalized.split()
        if not tokens:
            return (self._build_chunk(body=normalized, input_id=input_id, index=0),)

        chunks: list[ProblemChunk] = []
        start = 0
        while start < len(tokens):
            end = min(start + self._window_tokens, len(tokens))
            body = " ".join(tokens[start:end]).strip()
            if is_non_empty_chunk_text(body):
                chunks.append(self._build_chunk(body=body, input_id=input_id, index=len(chunks)))
            if end == len(tokens):
                break
            start = max(end - self._overlap_tokens, start + 1)

        if chunks:
            return tuple(chunks)
        return (self._build_chunk(body=normalized, input_id=input_id, index=0),)

    def _build_chunk(self, *, body: str, input_id: str, index: int) -> ProblemChunk:
        chunk_id = f"{input_id}:fallback:{index}"
        return ProblemChunk(
            id=chunk_id,
            chunk_id=chunk_id,
            problem_id=input_id,
            input_id=input_id,
            kind="fallback",
            text=body,
            display_text=body,
            index=index,
            metadata={"chunker": "generic_fallback", "labOnly": True},
        )

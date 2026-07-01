from __future__ import annotations

from ..contracts import ProblemChunk, RawProblem
from .structured_problem import StructuredProblemChunker

SUPPORTED_RUNTIME_TYPES = ("structured_problem",)


class ChunkingRouter:
    def __init__(self) -> None:
        self._structured_problem_chunker = StructuredProblemChunker()

    @staticmethod
    def supported_runtime_types() -> tuple[str, ...]:
        return SUPPORTED_RUNTIME_TYPES

    def chunk_problem(
        self,
        problem: RawProblem,
        *,
        runtime_type: str = "structured_problem",
    ) -> tuple[ProblemChunk, ...]:
        if runtime_type not in SUPPORTED_RUNTIME_TYPES:
            raise ValueError(f"unsupported chunking runtime type: {runtime_type}")
        return self._structured_problem_chunker.chunk(problem)

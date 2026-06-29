from __future__ import annotations

from ..contracts import ProblemChunk, RawProblem
from .structured_problem import StructuredProblemChunker


class ChunkingRouter:
    def __init__(self) -> None:
        self._chunkers = {
            "structured_problem": StructuredProblemChunker(),
        }

    @staticmethod
    def supported_runtime_types() -> tuple[str, ...]:
        return ("structured_problem",)

    def chunk_problem(
        self,
        problem: RawProblem,
        *,
        runtime_type: str = "structured_problem",
    ) -> tuple[ProblemChunk, ...]:
        chunker = self._chunkers.get(runtime_type)
        if chunker is None:
            raise ValueError(f"unsupported chunking runtime type: {runtime_type}")
        return chunker.chunk(problem)

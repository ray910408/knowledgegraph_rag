from __future__ import annotations

from typing import Any

from ..contracts import ProblemChunk, RawProblem
from .search_text import build_chunk_search_text, is_non_empty_chunk_text


class StructuredProblemChunker:
    def chunk(self, problem: RawProblem) -> tuple[ProblemChunk, ...]:
        chunks: list[ProblemChunk] = []

        self._append_chunk(
            chunks,
            problem,
            kind="problem_card",
            text=self._problem_card_text(problem),
        )
        self._append_chunk(chunks, problem, kind="statement", text=problem.statement)
        self._append_chunk(
            chunks,
            problem,
            kind="constraints",
            text="\n".join(problem.constraints),
        )
        self._append_chunk(
            chunks,
            problem,
            kind="examples",
            text=self._examples_text(problem.examples),
        )
        self._append_chunk(
            chunks,
            problem,
            kind="hints",
            text="\n".join(hint for hint in problem.solution_hints if is_non_empty_chunk_text(hint)),
        )
        self._append_chunk(
            chunks,
            problem,
            kind="solution",
            text=self._solution_text(problem),
        )
        self._append_chunk(
            chunks,
            problem,
            kind="common_mistakes",
            text=self._common_mistakes_text(problem.metadata),
        )

        return tuple(chunks)

    def _append_chunk(
        self,
        chunks: list[ProblemChunk],
        problem: RawProblem,
        *,
        kind: str,
        text: str,
    ) -> None:
        if not is_non_empty_chunk_text(text):
            return
        index = sum(1 for chunk in chunks if chunk.kind == kind)
        chunks.append(
            ProblemChunk(
                id=f"{problem.id}:{kind}:{index}",
                problem_id=problem.id,
                kind=kind,
                text=text,
                display_text=text,
                search_text=build_chunk_search_text(
                    problem_id=problem.id,
                    source=problem.source,
                    source_id=problem.source_id,
                    title=problem.title,
                    problem_type=problem.problem_type,
                    concepts=problem.concepts,
                    display_text=text,
                ),
                index=index,
                concepts=problem.concepts,
                metadata={
                    "source": problem.source,
                    "sourceId": problem.source_id,
                    "title": problem.title,
                    "problemType": problem.problem_type,
                },
                answer=problem.answer,
                solution_hints=problem.solution_hints,
                difficulty=problem.difficulty or self._metadata_difficulty(problem.metadata),
                constraints=problem.constraints,
                examples=problem.examples,
                editorial=problem.editorial,
                source=problem.source,
                source_id=problem.source_id,
                title=problem.title,
                problem_type=problem.problem_type,
            )
        )

    def _problem_card_text(self, problem: RawProblem) -> str:
        parts = [
            f"{problem.source} {problem.source_id}",
            problem.title,
            problem.problem_type,
        ]
        difficulty = problem.difficulty or self._metadata_difficulty(problem.metadata)
        if is_non_empty_chunk_text(difficulty or ""):
            parts.append(f"Difficulty: {difficulty}")
        if problem.tags:
            parts.append(f"Tags: {', '.join(problem.tags)}")
        return " | ".join(parts)

    def _examples_text(self, examples: tuple[dict[str, Any], ...]) -> str:
        blocks: list[str] = []
        for index, example in enumerate(examples, start=1):
            lines = [f"Example {index}"]
            for label, key in (("Input", "input"), ("Output", "output"), ("Explanation", "explanation")):
                value = str(example.get(key, "")).strip()
                if is_non_empty_chunk_text(value):
                    lines.append(f"{label}: {value}")
            block = "\n".join(lines)
            if is_non_empty_chunk_text(block.replace(f"Example {index}", "", 1)):
                blocks.append(block)
        return "\n\n".join(blocks)

    def _solution_text(self, problem: RawProblem) -> str:
        parts: list[str] = []
        if is_non_empty_chunk_text(problem.answer):
            parts.append(problem.answer)
        if is_non_empty_chunk_text(problem.editorial or ""):
            parts.append(str(problem.editorial))
        return "\n\n".join(parts)

    def _common_mistakes_text(self, metadata: dict[str, Any]) -> str:
        if str(metadata.get("commonMistakesSource", "")).strip().lower() == "template":
            return ""
        values = metadata.get("commonMistakes")
        if not isinstance(values, (list, tuple)):
            return ""
        mistakes = [str(value).strip() for value in values if is_non_empty_chunk_text(str(value))]
        return "\n".join(mistakes)

    def _metadata_difficulty(self, metadata: dict[str, Any]) -> str | None:
        value = metadata.get("difficulty")
        return None if value is None else str(value)

from __future__ import annotations

import re

from ..contracts import ProblemChunk
from .search_text import is_non_empty_chunk_text

_HEADING_ALIASES: dict[str, tuple[str, ...]] = {
    "statement": ("description", "problem statement", "statement", "題目敘述", "題意"),
    "input": ("input", "輸入"),
    "output": ("output", "輸出"),
    "sample_input": ("sample input", "input example"),
    "sample_output": ("sample output", "output example"),
    "examples": ("examples", "example", "範例", "範例輸入輸出"),
}


class ProblemStatementChunker:
    def chunk(self, text: str, *, input_id: str) -> tuple[ProblemChunk, ...]:
        normalized = text.strip()
        if not is_non_empty_chunk_text(normalized):
            return ()

        sections = self._section_chunks(normalized, input_id=input_id)
        if sections:
            return sections
        return self._paragraph_chunks(normalized, input_id=input_id)

    def _section_chunks(self, text: str, *, input_id: str) -> tuple[ProblemChunk, ...]:
        sample_input_lines: list[str] = []
        sample_output_lines: list[str] = []
        collected: list[tuple[str, str]] = []

        current_heading = "statement"
        current_lines: list[str] = []
        matched_heading = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            heading = self._match_heading(line)
            if heading is None:
                current_lines.append(raw_line)
                continue

            matched_heading = True
            self._flush_section(
                heading=current_heading,
                lines=current_lines,
                collected=collected,
                sample_input_lines=sample_input_lines,
                sample_output_lines=sample_output_lines,
            )
            current_heading = heading
            current_lines = []

        self._flush_section(
            heading=current_heading,
            lines=current_lines,
            collected=collected,
            sample_input_lines=sample_input_lines,
            sample_output_lines=sample_output_lines,
        )

        if not matched_heading:
            return ()

        if sample_input_lines or sample_output_lines:
            example_parts: list[str] = []
            if sample_input_lines:
                example_parts.append("Sample Input\n" + "\n".join(sample_input_lines).strip())
            if sample_output_lines:
                example_parts.append("Sample Output\n" + "\n".join(sample_output_lines).strip())
            collected.append(("examples", "\n\n".join(part for part in example_parts if is_non_empty_chunk_text(part))))

        chunks: list[ProblemChunk] = []
        for kind, body in collected:
            if not is_non_empty_chunk_text(body):
                continue
            index = sum(1 for chunk in chunks if chunk.kind == kind)
            chunks.append(self._build_chunk(kind=kind, body=body.strip(), input_id=input_id, index=index))
        return tuple(chunks)

    def _paragraph_chunks(self, text: str, *, input_id: str) -> tuple[ProblemChunk, ...]:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n+", text)
            if is_non_empty_chunk_text(paragraph)
        ]
        if not paragraphs:
            paragraphs = [text.strip()]
        return tuple(
            self._build_chunk(kind="statement", body=paragraph, input_id=input_id, index=index)
            for index, paragraph in enumerate(paragraphs)
        )

    def _flush_section(
        self,
        *,
        heading: str,
        lines: list[str],
        collected: list[tuple[str, str]],
        sample_input_lines: list[str],
        sample_output_lines: list[str],
    ) -> None:
        body = "\n".join(lines).strip()
        if not is_non_empty_chunk_text(body):
            return
        if heading == "sample_input":
            sample_input_lines.extend(body.splitlines())
            return
        if heading == "sample_output":
            sample_output_lines.extend(body.splitlines())
            return
        collected.append((heading, body))

    def _match_heading(self, line: str) -> str | None:
        normalized = re.sub(r"[:：]\s*$", "", line).strip().casefold()
        for kind, aliases in _HEADING_ALIASES.items():
            if normalized in aliases:
                return kind
        return None

    def _build_chunk(self, *, kind: str, body: str, input_id: str, index: int) -> ProblemChunk:
        chunk_id = f"{input_id}:{kind}:{index}"
        return ProblemChunk(
            id=chunk_id,
            chunk_id=chunk_id,
            problem_id=input_id,
            input_id=input_id,
            kind=kind,
            text=body,
            display_text=body,
            index=index,
            metadata={"chunker": "problem_statement", "labOnly": True},
        )

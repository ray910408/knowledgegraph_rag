from __future__ import annotations

import re

from ..contracts import ProblemChunk
from .optional_parser import OptionalStructuredParser
from .search_text import is_non_empty_chunk_text

_CPP_PATTERN = re.compile(
    r"(?m)^\s*(?:template\s*<[^>]+>\s*)?(?:[\w:<>~*&]+\s+)+(?P<name>[A-Za-z_]\w*)\s*\([^)]*\)\s*\{"
)
_PYTHON_PATTERN = re.compile(r"(?m)^\s*def\s+(?P<name>[A-Za-z_]\w*)\s*\([^)]*\)\s*:")
_JAVASCRIPT_PATTERN = re.compile(
    r"(?m)^\s*(?:function\s+)?(?P<name>[A-Za-z_]\w*)\s*=\s*\([^)]*\)\s*=>|^\s*function\s+(?P<fn>[A-Za-z_]\w*)\s*\("
)
_JAVA_PATTERN = re.compile(
    r"(?m)^\s*(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+(?P<name>[A-Za-z_]\w*)\s*\([^)]*\)\s*\{"
)


class CodeChunker:
    def __init__(
        self,
        *,
        parser: OptionalStructuredParser | None = None,
        window_lines: int = 20,
        overlap_lines: int = 5,
    ) -> None:
        self._parser = parser
        self._window_lines = max(window_lines, overlap_lines + 1)
        self._overlap_lines = overlap_lines

    def detect_language(self, text: str) -> str:
        if "#include" in text or "using namespace std" in text or re.search(r"\bint\s+main\s*\(", text):
            return "cpp"
        if re.search(r"(?m)^\s*def\s+[A-Za-z_]\w*\s*\(", text):
            return "python"
        if "function " in text or "=>" in text or "console.log" in text:
            return "javascript"
        if "public class " in text or "public static void main" in text:
            return "java"
        return "text"

    def chunk(self, text: str, *, input_id: str) -> tuple[ProblemChunk, ...]:
        normalized = text.strip()
        if not is_non_empty_chunk_text(normalized):
            return ()

        language = self.detect_language(normalized)

        parser_chunks = self._parser_chunks(normalized, input_id=input_id, language=language)
        if parser_chunks:
            return parser_chunks

        function_chunks = self._function_chunks(normalized, input_id=input_id, language=language)
        if function_chunks:
            return function_chunks

        blank_line_chunks = self._blank_line_chunks(normalized, input_id=input_id, language=language)
        if blank_line_chunks:
            return blank_line_chunks

        return self._line_window_chunks(normalized, input_id=input_id, language=language)

    def _parser_chunks(self, text: str, *, input_id: str, language: str) -> tuple[ProblemChunk, ...]:
        if self._parser is None:
            return ()
        parsed = self._parser.parse(text, language_hint=None if language == "text" else language)
        if not parsed:
            return ()

        raw_chunks = parsed.get("chunks")
        if not isinstance(raw_chunks, list):
            return ()

        chunks: list[ProblemChunk] = []
        for raw_chunk in raw_chunks:
            if not isinstance(raw_chunk, dict):
                continue
            body = str(raw_chunk.get("text", "")).strip()
            if not is_non_empty_chunk_text(body):
                continue
            kind = str(raw_chunk.get("kind", "code_block"))
            metadata = {"chunker": "code", "labOnly": True, "language": language}
            symbol_name = str(raw_chunk.get("symbolName", "")).strip()
            if symbol_name:
                metadata["symbolName"] = symbol_name
            chunks.append(
                self._build_chunk(
                    kind=kind,
                    body=body,
                    input_id=input_id,
                    index=len(chunks),
                    metadata=metadata,
                )
            )
        return tuple(chunks)

    def _function_chunks(self, text: str, *, input_id: str, language: str) -> tuple[ProblemChunk, ...]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if is_non_empty_chunk_text(block)]
        if not blocks:
            return ()

        chunks: list[ProblemChunk] = []
        for block in blocks:
            symbol_name = self._match_symbol_name(block, language=language)
            if not symbol_name:
                continue
            chunks.append(
                self._build_chunk(
                    kind="code_function",
                    body=block,
                    input_id=input_id,
                    index=len(chunks),
                    metadata={
                        "chunker": "code",
                        "labOnly": True,
                        "language": language,
                        "symbolName": symbol_name,
                    },
                )
            )
        return tuple(chunks)

    def _blank_line_chunks(self, text: str, *, input_id: str, language: str) -> tuple[ProblemChunk, ...]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if is_non_empty_chunk_text(block)]
        if not blocks:
            return ()
        if all(len([line for line in block.splitlines() if is_non_empty_chunk_text(line)]) <= 1 for block in blocks):
            return ()
        return tuple(
            self._build_chunk(
                kind="code_block",
                body=block,
                input_id=input_id,
                index=index,
                metadata={"chunker": "code", "labOnly": True, "language": language},
            )
            for index, block in enumerate(blocks)
        )

    def _line_window_chunks(self, text: str, *, input_id: str, language: str) -> tuple[ProblemChunk, ...]:
        lines = [line for line in text.splitlines() if is_non_empty_chunk_text(line)]
        if not lines:
            return ()

        chunks: list[ProblemChunk] = []
        start = 0
        while start < len(lines):
            end = min(start + self._window_lines, len(lines))
            body = "\n".join(lines[start:end]).strip()
            if is_non_empty_chunk_text(body):
                chunks.append(
                    self._build_chunk(
                        kind="code_window",
                        body=body,
                        input_id=input_id,
                        index=len(chunks),
                        metadata={
                            "chunker": "code",
                            "labOnly": True,
                            "language": language,
                            "lineStart": start + 1,
                            "lineEnd": end,
                        },
                    )
                )
            if end == len(lines):
                break
            start = max(end - self._overlap_lines, start + 1)
        return tuple(chunks)

    def _match_symbol_name(self, block: str, *, language: str) -> str | None:
        pattern = {
            "cpp": _CPP_PATTERN,
            "python": _PYTHON_PATTERN,
            "javascript": _JAVASCRIPT_PATTERN,
            "java": _JAVA_PATTERN,
        }.get(language)
        if pattern is None:
            return None
        match = pattern.search(block)
        if match is None:
            return None
        return (match.groupdict().get("name") or match.groupdict().get("fn") or "").strip() or None

    def _build_chunk(
        self,
        *,
        kind: str,
        body: str,
        input_id: str,
        index: int,
        metadata: dict[str, object],
    ) -> ProblemChunk:
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
            metadata=metadata,
        )

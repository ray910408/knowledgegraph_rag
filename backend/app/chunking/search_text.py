from __future__ import annotations

from ..query_language import build_search_text


def is_non_empty_chunk_text(value: str) -> bool:
    return bool(value.strip())


def build_chunk_search_text(
    *,
    problem_id: str,
    source: str,
    source_id: str,
    title: str,
    problem_type: str,
    concepts: tuple[str, ...],
    display_text: str,
) -> str:
    if not is_non_empty_chunk_text(display_text):
        return ""
    return " ".join(
        build_search_text(
            problem_id=problem_id,
            source=source,
            source_id=source_id,
            title=title,
            problem_type=problem_type,
            concepts=concepts,
            display_text=display_text,
        ).split()
    )

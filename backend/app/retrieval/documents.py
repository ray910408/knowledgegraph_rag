from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pipeline import JsonMap, RetrievalDocument


class ProcessedProblemDocumentLoaderError(RuntimeError):
    pass


class ProcessedProblemDocumentLoader:
    _required_fields = (
        "id",
        "source",
        "sourceId",
        "title",
        "problemType",
        "statement",
        "answer",
    )

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> tuple[RetrievalDocument, ...]:
        if not self._path.exists():
            raise ProcessedProblemDocumentLoaderError(
                f"processed problems not found: {self._path}"
            )

        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProcessedProblemDocumentLoaderError(
                f"processed problems payload is not valid JSON: {self._path}"
            ) from exc

        raw_problems = self._unwrap_payload(payload)
        if not raw_problems:
            raise ProcessedProblemDocumentLoaderError(
                f"processed problems payload is empty: {self._path}"
            )

        documents = tuple(
            self._document_from_mapping(raw_problem, index=index)
            for index, raw_problem in enumerate(raw_problems)
        )
        if not documents:
            raise ProcessedProblemDocumentLoaderError(
                f"processed problems produced no retrieval documents: {self._path}"
            )
        return documents

    def _unwrap_payload(self, payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "problems" in payload:
            problems = payload["problems"]
            if isinstance(problems, list):
                return problems
        raise ProcessedProblemDocumentLoaderError(
            "processed problems payload must be a list or a mapping with a problems list"
        )

    def _document_from_mapping(self, raw: Any, *, index: int) -> RetrievalDocument:
        if not isinstance(raw, dict):
            raise ProcessedProblemDocumentLoaderError(
                f"processed problem at index {index} must be an object"
            )

        missing = [
            field for field in self._required_fields if field not in raw or raw[field] is None
        ]
        if missing:
            raise ProcessedProblemDocumentLoaderError(
                f"processed problem at index {index} is missing required fields: "
                f"{', '.join(missing)}"
            )

        metadata = _metadata(raw.get("metadata"), field="metadata", index=index)
        difficulty = raw.get("difficulty")
        if difficulty is None:
            difficulty = metadata.get("difficulty")

        return RetrievalDocument(
            id=_required_scalar(raw, field="id", index=index),
            source=_required_scalar(raw, field="source", index=index),
            source_id=_required_scalar(raw, field="sourceId", index=index),
            title=_required_scalar(raw, field="title", index=index),
            text=_required_scalar(raw, field="statement", index=index),
            answer=_required_scalar(raw, field="answer", index=index),
            concepts=_tuple_of_str(raw.get("concepts"), field="concepts", index=index),
            problem_type=_required_scalar(raw, field="problemType", index=index),
            solution_hints=_tuple_of_str(
                raw.get("solutionHints"), field="solutionHints", index=index
            ),
            difficulty=_optional_scalar(difficulty, field="difficulty", index=index),
            constraints=_tuple_of_str(
                raw.get("constraints"), field="constraints", index=index
            ),
            examples=_tuple_of_mapping(raw.get("examples"), field="examples", index=index),
            editorial=_optional_scalar(raw.get("editorial"), field="editorial", index=index),
            metadata=metadata,
        )


_CONTAINER_TYPES = (dict, list, tuple, set)
_SEQUENCE_TYPES = (list, tuple)


def _required_scalar(raw: dict[str, Any], *, field: str, index: int) -> str:
    value = raw[field]
    if _is_container(value):
        raise _field_error(index=index, field=field, reason="must be a scalar value")
    return str(value)


def _optional_scalar(value: Any, *, field: str, index: int) -> str:
    if value is None:
        return ""
    if _is_container(value):
        raise _field_error(index=index, field=field, reason="must be a scalar value")
    return str(value)


def _metadata(value: Any, *, field: str, index: int) -> JsonMap:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _field_error(index=index, field=field, reason="must be an object")
    return dict(value)


def _tuple_of_str(value: Any, *, field: str, index: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, _SEQUENCE_TYPES):
        raise _field_error(index=index, field=field, reason="must be a string or list")

    values: list[str] = []
    for item_index, item in enumerate(value):
        if _is_container(item):
            raise _field_error(
                index=index,
                field=field,
                reason=f"item {item_index} must be a scalar value",
            )
        values.append(str(item))
    return tuple(values)


def _tuple_of_mapping(value: Any, *, field: str, index: int) -> tuple[JsonMap, ...]:
    if value is None:
        return ()
    if not isinstance(value, _SEQUENCE_TYPES):
        raise _field_error(index=index, field=field, reason="must be a list of objects")

    values: list[JsonMap] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, dict):
            raise _field_error(
                index=index,
                field=field,
                reason=f"item {item_index} must be an object",
            )
        values.append(dict(item))
    return tuple(values)


def _is_container(value: Any) -> bool:
    return isinstance(value, _CONTAINER_TYPES)


def _field_error(*, index: int, field: str, reason: str) -> ProcessedProblemDocumentLoaderError:
    return ProcessedProblemDocumentLoaderError(
        f"processed problem at index {index} field {field} {reason}"
    )

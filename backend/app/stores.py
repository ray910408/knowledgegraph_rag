from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from .contracts import EntityRecord, RelationRecord


JsonMap = dict[str, Any]


@dataclass(frozen=True)
class SearchCandidate:
    id: str
    score: float
    payload: JsonMap = field(default_factory=dict)


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: tuple[float, ...]
    payload: JsonMap = field(default_factory=dict)


@dataclass(frozen=True)
class BM25Document:
    id: str
    text: str
    payload: JsonMap = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, records: Sequence[VectorRecord]) -> None:
        ...

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        filters: JsonMap | None = None,
    ) -> tuple[SearchCandidate, ...]:
        ...


@runtime_checkable
class BM25Store(Protocol):
    def index_documents(self, documents: Sequence[BM25Document]) -> None:
        ...

    def search(self, query: str, *, top_k: int) -> tuple[SearchCandidate, ...]:
        ...


@runtime_checkable
class GraphStore(Protocol):
    def upsert_entities(self, entities: Sequence[EntityRecord]) -> None:
        ...

    def upsert_relations(self, relations: Sequence[RelationRecord]) -> None:
        ...

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        *,
        max_hops: int = 3,
    ) -> tuple[JsonMap, ...]:
        ...

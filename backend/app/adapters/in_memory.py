from __future__ import annotations

import math
from collections import deque
from typing import Sequence

from ..contracts import EntityRecord, RelationRecord
from ..query_language import shared_multilingual_tokens
from ..stores import BM25Document, SearchCandidate, VectorRecord


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        for record in records:
            self._records[record.id] = record

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> tuple[SearchCandidate, ...]:
        candidates = [
            SearchCandidate(
                id=record.id,
                score=round(_cosine(query_vector, record.vector), 6),
                payload=dict(record.payload),
            )
            for record in self._records.values()
            if _matches_filters(record.payload, filters)
        ]
        return tuple(sorted(candidates, key=lambda item: (-item.score, item.id))[:top_k])


class InMemoryBM25Store:
    def __init__(self) -> None:
        self._documents: dict[str, BM25Document] = {}

    def index_documents(self, documents: Sequence[BM25Document]) -> None:
        for document in documents:
            self._documents[document.id] = document

    def search(self, query: str, *, top_k: int) -> tuple[SearchCandidate, ...]:
        query_terms = set(_tokens(query))
        candidates: list[SearchCandidate] = []
        for document in self._documents.values():
            terms = _tokens(document.text)
            overlap = query_terms & set(terms)
            score = 0.0 if not query_terms else len(overlap) / len(query_terms)
            score += sum(1 for term in terms if term in query_terms) / max(len(terms), 1)
            candidates.append(
                SearchCandidate(id=document.id, score=round(score, 6), payload=dict(document.payload))
            )
        return tuple(sorted(candidates, key=lambda item: (-item.score, item.id))[:top_k])


class InMemoryGraphStore:
    def __init__(self) -> None:
        self._entities: dict[str, EntityRecord] = {}
        self._relations: list[RelationRecord] = []

    def upsert_entities(self, entities: Sequence[EntityRecord]) -> None:
        for entity in entities:
            self._entities[entity.id] = entity

    def upsert_relations(self, relations: Sequence[RelationRecord]) -> None:
        existing = {relation.id: relation for relation in self._relations}
        for relation in relations:
            existing[relation.id] = relation
        self._relations = list(existing.values())

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        *,
        max_hops: int = 3,
    ) -> tuple[dict[str, object], ...]:
        adjacency: dict[str, list[RelationRecord]] = {}
        for relation in self._relations:
            adjacency.setdefault(relation.source_id, []).append(relation)

        queue = deque([(source_id, [source_id], [])])
        paths: list[dict[str, object]] = []
        while queue:
            current, nodes, relations = queue.popleft()
            if len(relations) >= max_hops:
                continue
            for relation in adjacency.get(current, []):
                next_nodes = [*nodes, relation.target_id]
                next_relations = [*relations, relation.type]
                if relation.target_id == target_id:
                    paths.append(
                        {
                            "nodes": next_nodes,
                            "relations": next_relations,
                            "score": round(relation.weight, 6),
                        }
                    )
                    continue
                if relation.target_id not in nodes:
                    queue.append((relation.target_id, next_nodes, next_relations))
        return tuple(paths)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _matches_filters(payload: dict[str, object], filters: dict[str, object] | None) -> bool:
    if not filters:
        return True
    return all(payload.get(key) == value for key, value in filters.items())


def _tokens(text: str) -> tuple[str, ...]:
    return shared_multilingual_tokens(text)

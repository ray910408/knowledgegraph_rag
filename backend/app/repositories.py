from __future__ import annotations

from collections import defaultdict, deque
import re
from typing import Iterable, Protocol

from .domain import Concept, EvidencePath, Problem, Relationship, VectorCandidate


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> frozenset[str]:
    return frozenset(TOKEN_PATTERN.findall(text.lower()))


class GraphRepository(Protocol):
    def add_problem(self, problem: Problem) -> None:
        ...

    def add_concept(self, concept: Concept) -> None:
        ...

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        ...

    def get_problem(self, problem_id: str) -> Problem | None:
        ...

    def list_problems(self) -> tuple[Problem, ...]:
        ...

    def get_concept(self, concept_id: str) -> Concept | None:
        ...

    def list_concepts(self) -> tuple[Concept, ...]:
        ...

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        excluded_relation_types: Iterable[str] = (),
    ) -> tuple[EvidencePath, ...]:
        ...

    def is_excluded(
        self,
        source_ids: Iterable[str],
        target_ids: Iterable[str],
    ) -> bool:
        ...


class VectorRepository(Protocol):
    def upsert_problem(self, problem: Problem) -> None:
        ...

    def query_similar(
        self,
        text: str,
        top_k: int,
        exclude_ids: Iterable[str] = (),
    ) -> tuple[VectorCandidate, ...]:
        ...


class InMemoryGraphRepository:
    def __init__(self) -> None:
        self._problems: dict[str, Problem] = {}
        self._concepts: dict[str, Concept] = {}
        self._relationships: list[Relationship] = []
        self._adjacency: dict[str, list[Relationship]] = defaultdict(list)

    def add_problem(self, problem: Problem) -> None:
        self._problems[problem.id] = problem

    def add_concept(self, concept: Concept) -> None:
        self._concepts[concept.id] = concept

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            type=relation_type.upper(),
            weight=weight,
        )
        self._relationships.append(relationship)
        self._adjacency[source_id].append(relationship)

    def link_problem_concept(
        self,
        problem_id: str,
        concept_id: str,
        relation_type: str = "USES",
        weight: float = 1.0,
    ) -> None:
        self.add_relationship(problem_id, concept_id, relation_type, weight)

    def get_problem(self, problem_id: str) -> Problem | None:
        return self._problems.get(problem_id)

    def list_problems(self) -> tuple[Problem, ...]:
        return tuple(self._problems.values())

    def get_concept(self, concept_id: str) -> Concept | None:
        return self._concepts.get(concept_id)

    def list_concepts(self) -> tuple[Concept, ...]:
        return tuple(self._concepts.values())

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        excluded_relation_types: Iterable[str] = (),
    ) -> tuple[EvidencePath, ...]:
        excluded = {item.upper() for item in excluded_relation_types}
        if max_hops < 1:
            return ()

        found: list[EvidencePath] = []
        queue = deque([(source_id, (source_id,), (), 1.0)])

        while queue:
            current_id, nodes, relations, score = queue.popleft()
            if len(relations) >= max_hops:
                continue

            for relationship in self._adjacency.get(current_id, ()):
                if relationship.type in excluded:
                    continue
                if relationship.target_id in nodes:
                    continue

                next_nodes = (*nodes, relationship.target_id)
                next_relations = (*relations, relationship.type)
                next_score = score * max(relationship.weight, 0.0)

                if relationship.target_id == target_id:
                    found.append(
                        EvidencePath(
                            nodes=next_nodes,
                            relations=next_relations,
                            score=_path_score(next_score, len(next_relations)),
                            rationale=" -> ".join(next_relations),
                        )
                    )
                    continue

                queue.append(
                    (
                        relationship.target_id,
                        next_nodes,
                        next_relations,
                        next_score,
                    )
                )

        return tuple(sorted(found, key=lambda item: item.score, reverse=True))

    def is_excluded(
        self,
        source_ids: Iterable[str],
        target_ids: Iterable[str],
    ) -> bool:
        sources = set(source_ids)
        targets = set(target_ids)
        for relationship in self._relationships:
            if relationship.type != "EXCLUDES":
                continue
            forward = relationship.source_id in sources and relationship.target_id in targets
            reverse = relationship.source_id in targets and relationship.target_id in sources
            if forward or reverse:
                return True
        return False


class InMemoryVectorRepository:
    def __init__(self) -> None:
        self._problems: dict[str, Problem] = {}
        self._tokens: dict[str, frozenset[str]] = {}

    def upsert_problem(self, problem: Problem) -> None:
        self._problems[problem.id] = problem
        self._tokens[problem.id] = tokenize(problem.text)

    def query_similar(
        self,
        text: str,
        top_k: int,
        exclude_ids: Iterable[str] = (),
    ) -> tuple[VectorCandidate, ...]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_tokens = tokenize(text)
        excluded = set(exclude_ids)
        candidates: list[VectorCandidate] = []

        for problem_id, problem in self._problems.items():
            if problem_id in excluded:
                continue
            score = _token_similarity(query_tokens, self._tokens[problem_id])
            if score <= 0:
                continue
            candidates.append(VectorCandidate(problem=problem, score=score))

        candidates.sort(key=lambda item: (-item.score, item.problem.id))
        return tuple(candidates[:top_k])


def _token_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    return overlap / len(left | right)


def _path_score(weight: float, hops: int) -> float:
    if hops <= 0:
        return 0.0
    return weight / hops

from __future__ import annotations

from ..domain import EvidenceBundle, EvidencePath, Problem, Recommendation
from ..repositories import GraphRepository, VectorRepository, tokenize


class HybridRetrievalService:
    def __init__(
        self,
        graph_repository: GraphRepository,
        vector_repository: VectorRepository,
        vector_weight: float = 0.35,
        graph_weight: float = 0.35,
        concept_weight: float = 0.30,
    ) -> None:
        self._graph = graph_repository
        self._vectors = vector_repository
        self._vector_weight = vector_weight
        self._graph_weight = graph_weight
        self._concept_weight = concept_weight

    def recommend(
        self,
        *,
        problem_text: str | None = None,
        problem_id: str | None = None,
        top_k: int = 5,
    ) -> EvidenceBundle:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not problem_id and not problem_text:
            raise ValueError("problem_id or problem_text is required")

        query_problem = self._resolve_query_problem(problem_id, problem_text)
        query_concept_ids = set(query_problem.concept_ids) | self._infer_concepts(
            query_problem.text
        )
        candidate_limit = max(top_k * 4, top_k, 1)
        excluded_ids = {query_problem.id} if problem_id else set()
        vector_candidates = self._vectors.query_similar(
            query_problem.text,
            top_k=candidate_limit,
            exclude_ids=excluded_ids,
        )

        recommendations: list[Recommendation] = []
        query_node_ids = {query_problem.id, *query_concept_ids}
        candidate_records: list[tuple[Problem, float, tuple[EvidencePath, ...] | None]] = [
            (candidate.problem, candidate.score, None) for candidate in vector_candidates
        ]
        seen_candidate_ids = {candidate.problem.id for candidate in vector_candidates}

        for problem in self._graph.list_problems():
            if problem.id in seen_candidate_ids or problem.id in excluded_ids:
                continue
            evidence_paths = self._evidence_paths(
                query_problem,
                problem,
                problem_id,
                query_concept_ids,
            )
            concept_score = self._concept_score(query_concept_ids, problem.concept_ids)
            if evidence_paths or concept_score > 0:
                candidate_records.append((problem, 0.0, evidence_paths))

        for problem, vector_score, precomputed_paths in candidate_records:
            target_node_ids = {problem.id, *problem.concept_ids}
            if self._graph.is_excluded(query_node_ids, target_node_ids):
                continue

            concept_matches = self._concept_matches(query_concept_ids, problem)
            concept_score = self._concept_score(query_concept_ids, problem.concept_ids)
            evidence_paths = (
                precomputed_paths
                if precomputed_paths is not None
                else self._evidence_paths(
                    query_problem,
                    problem,
                    problem_id,
                    query_concept_ids,
                )
            )
            graph_score = max((path.score for path in evidence_paths), default=0.0)
            total_score = (
                self._vector_weight * vector_score
                + self._graph_weight * graph_score
                + self._concept_weight * concept_score
            )

            recommendations.append(
                Recommendation(
                    problem_id=problem.id,
                    title=problem.title,
                    score=round(total_score, 6),
                    vector_score=round(vector_score, 6),
                    graph_score=round(graph_score, 6),
                    concept_match_score=round(concept_score, 6),
                    concept_matches=concept_matches,
                    evidence_paths=evidence_paths,
                )
            )

        recommendations.sort(
            key=lambda item: (
                -item.score,
                -item.graph_score,
                -item.concept_match_score,
                item.problem_id,
            )
        )
        return EvidenceBundle(
            query_problem=query_problem,
            recommendations=tuple(recommendations[:top_k]),
        )

    def _resolve_query_problem(
        self,
        problem_id: str | None,
        problem_text: str | None,
    ) -> Problem:
        if problem_id:
            problem = self._graph.get_problem(problem_id)
            if problem is None:
                raise ValueError(f"unknown problem_id: {problem_id}")
            return problem

        assert problem_text is not None
        return Problem(
            id="query",
            title="Ad hoc query",
            text=problem_text,
            concept_ids=tuple(sorted(self._infer_concepts(problem_text))),
        )

    def _infer_concepts(self, text: str) -> set[str]:
        text_tokens = tokenize(text)
        matches: set[str] = set()
        for concept in self._graph.list_concepts():
            names = (concept.name, *concept.aliases)
            if any(_phrase_matches(text_tokens, name) for name in names):
                matches.add(concept.id)
        return matches

    def _concept_score(
        self,
        query_concept_ids: set[str],
        candidate_concept_ids: tuple[str, ...],
    ) -> float:
        if not query_concept_ids:
            return 0.0
        overlap = query_concept_ids & set(candidate_concept_ids)
        return len(overlap) / len(query_concept_ids)

    def _concept_matches(
        self,
        query_concept_ids: set[str],
        problem: Problem,
    ) -> tuple[str, ...]:
        names: list[str] = []
        for concept_id in sorted(query_concept_ids & set(problem.concept_ids)):
            concept = self._graph.get_concept(concept_id)
            names.append(concept.name if concept else concept_id)
        return tuple(names)

    def _evidence_paths(
        self,
        query_problem: Problem,
        candidate: Problem,
        persisted_problem_id: str | None,
        query_concept_ids: set[str],
    ) -> tuple[EvidencePath, ...]:
        paths: list[EvidencePath] = []
        if persisted_problem_id:
            paths.extend(
                self._graph.find_paths(
                    query_problem.id,
                    candidate.id,
                    max_hops=3,
                    excluded_relation_types=("EXCLUDES",),
                )
            )

        concept_paths = self._shared_concept_paths(
            query_problem,
            candidate,
            query_concept_ids,
        )
        paths.extend(concept_paths)
        deduped = _dedupe_paths(paths)
        return tuple(sorted(deduped, key=lambda item: item.score, reverse=True))

    def _shared_concept_paths(
        self,
        query_problem: Problem,
        candidate: Problem,
        query_concept_ids: set[str],
    ) -> tuple[EvidencePath, ...]:
        shared = sorted(query_concept_ids & set(candidate.concept_ids))
        paths: list[EvidencePath] = []

        for concept_id in shared:
            concept = self._graph.get_concept(concept_id)
            concept_name = concept.name if concept else concept_id
            paths.append(
                EvidencePath(
                    nodes=(query_problem.id, concept_id, candidate.id),
                    relations=("MENTIONS", "USED_BY"),
                    score=0.5,
                    rationale=f"shared concept: {concept_name}",
                )
            )

        return tuple(paths)


def _dedupe_paths(paths: list[EvidencePath]) -> list[EvidencePath]:
    best_by_key: dict[tuple[tuple[str, ...], tuple[str, ...]], EvidencePath] = {}
    for path in paths:
        key = (path.nodes, path.relations)
        existing = best_by_key.get(key)
        if existing is None or path.score > existing.score:
            best_by_key[key] = path
    return list(best_by_key.values())


def _phrase_matches(text_tokens: frozenset[str], phrase: str) -> bool:
    phrase_tokens = tokenize(phrase)
    return bool(phrase_tokens) and phrase_tokens.issubset(text_tokens)

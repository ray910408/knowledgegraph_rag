from __future__ import annotations

from dataclasses import dataclass

from .demo import build_demo_repositories, recommend_demo_techniques


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    statement: str
    expected_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    mode: str
    top_k: int
    hit_rate: float
    cases: int


def run_fixture_evaluation(
    cases: tuple[EvaluationCase, ...],
    *,
    top_k: int = 3,
) -> tuple[EvaluationResult, ...]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    graph, _ = build_demo_repositories()
    results: list[EvaluationResult] = []
    for mode in ("vector", "graph", "hybrid"):
        hits = 0
        for case in cases:
            recommendations = recommend_demo_techniques(
                graph,
                problem_text=case.statement,
                top_k=top_k,
                mode=mode,
            )
            returned = {item.id for item in recommendations}
            if returned & set(case.expected_ids):
                hits += 1
        results.append(
            EvaluationResult(
                mode=mode,
                top_k=top_k,
                hit_rate=hits / len(cases) if cases else 0.0,
                cases=len(cases),
            )
        )
    return tuple(results)

from __future__ import annotations

from .domain import Concept, EvidencePath, Problem, TechniqueRecommendation
from .repositories import InMemoryGraphRepository, InMemoryVectorRepository, tokenize
from .services.llm import MockLLMProvider
from .services.retrieval import HybridRetrievalService


def build_demo_repositories() -> tuple[InMemoryGraphRepository, InMemoryVectorRepository]:
    graph = InMemoryGraphRepository()
    vectors = InMemoryVectorRepository()

    concepts = (
        Concept(
            id="range-sum",
            name="Range sum",
            kind="concept",
            aliases=("subarray sum", "contiguous subarray"),
            description="Queries or constraints over sums of contiguous ranges.",
        ),
        Concept(
            id="monotonic-window",
            name="Monotonic window",
            kind="concept",
            aliases=("positive numbers", "non-negative numbers", "non-negative"),
            description="A window property that moves predictably as endpoints change.",
        ),
        Concept(
            id="sliding-window",
            name="Sliding window",
            kind="pattern",
            aliases=("two pointers window",),
            description="Maintain a moving interval for contiguous constraints.",
            metadata={
                "pitfall": "Requires a monotonic condition such as non-negative values.",
            },
        ),
        Concept(
            id="prefix-sum",
            name="Prefix sum",
            kind="algorithm",
            aliases=("prefix sums",),
            description="Convert range sums into differences of accumulated prefixes.",
            metadata={
                "pitfall": "Hash-map initialization is easy to get wrong for zero prefix.",
            },
        ),
        Concept(
            id="hash-map",
            name="Hash map",
            kind="data_structure",
            aliases=("dictionary", "lookup table"),
            description="Store previously seen states for expected constant-time lookup.",
            metadata={"pitfall": "Memory usage can grow linearly with input size."},
        ),
        Concept(
            id="dijkstra",
            name="Dijkstra",
            kind="algorithm",
            aliases=("non-negative shortest path",),
            description="Shortest paths in weighted graphs with non-negative edges.",
            metadata={"pitfall": "Negative edge weights invalidate the greedy invariant."},
        ),
        Concept(
            id="negative-weights",
            name="Negative weights",
            kind="concept",
            aliases=("negative edge", "negative edges"),
            description="Edges with negative cost require algorithms other than Dijkstra.",
        ),
    )

    for concept in concepts:
        graph.add_concept(concept)

    examples = (
        Problem(
            id="demo-shortest-subarray",
            title="Shortest subarray at least target",
            text=(
                "Given non-negative integers and a target, find the shortest contiguous "
                "subarray whose sum is at least the target."
            ),
            concept_ids=("range-sum", "monotonic-window", "sliding-window"),
            metadata={"source": "demo"},
        ),
        Problem(
            id="demo-count-subarrays",
            title="Count subarrays with target sum",
            text=(
                "Given integers, count contiguous subarrays whose sum equals a target "
                "using prefix differences."
            ),
            concept_ids=("range-sum", "prefix-sum", "hash-map"),
            metadata={"source": "demo"},
        ),
        Problem(
            id="demo-network-delay",
            title="Network delay time",
            text="Find shortest paths from one node in a weighted graph with non-negative edges.",
            concept_ids=("dijkstra",),
            metadata={"source": "demo"},
        ),
    )

    for problem in examples:
        graph.add_problem(problem)
        vectors.upsert_problem(problem)

    graph.add_relationship("range-sum", "prefix-sum", "SOLVED_BY", 0.9)
    graph.add_relationship("range-sum", "hash-map", "USES", 0.7)
    graph.add_relationship("monotonic-window", "sliding-window", "HAS_PATTERN", 1.0)
    graph.add_relationship("prefix-sum", "hash-map", "USES", 0.8)
    graph.add_relationship("negative-weights", "dijkstra", "EXCLUDES", 1.0)
    graph.add_relationship("demo-shortest-subarray", "demo-count-subarrays", "SIMILAR_TO", 0.6)

    return graph, vectors


def build_demo_services() -> tuple[HybridRetrievalService, MockLLMProvider]:
    graph, vectors = build_demo_repositories()
    return HybridRetrievalService(graph, vectors), MockLLMProvider()


def recommend_demo_techniques(
    graph: InMemoryGraphRepository,
    *,
    problem_text: str,
    top_k: int,
    mode: str,
) -> tuple[TechniqueRecommendation, ...]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    query_terms = tokenize(problem_text)
    query_concepts = _infer_concepts(graph, problem_text)
    targets = [
        concept
        for concept in graph.list_concepts()
        if concept.kind in {"algorithm", "data_structure", "pattern"}
    ]

    recommendations: list[TechniqueRecommendation] = []
    for concept in targets:
        if graph.is_excluded(query_concepts, (concept.id,)):
            continue

        vector_score = _term_score(query_terms, concept)
        evidence_paths = _technique_paths(graph, query_concepts, concept.id)
        graph_score = max((path.score for path in evidence_paths), default=0.0)
        concept_score = 1.0 if concept.id in query_concepts else 0.0
        score = _weighted_score(
            mode,
            vector_score=vector_score,
            graph_score=graph_score,
            concept_score=concept_score,
        )

        if score <= 0:
            continue

        recommendations.append(
            TechniqueRecommendation(
                id=concept.id,
                kind=concept.kind,
                title=concept.name,
                score=round(score, 6),
                confidence=_confidence(score),
                summary=concept.description,
                fit_signals=_fit_signals(graph, query_concepts, concept.id),
                pitfalls=_pitfalls(concept),
                evidence_paths=evidence_paths,
            )
        )

    recommendations.sort(key=lambda item: (-item.score, item.kind, item.title))
    return tuple(recommendations[:top_k])


def _infer_concepts(graph: InMemoryGraphRepository, problem_text: str) -> tuple[str, ...]:
    text_tokens = tokenize(problem_text)
    inferred: list[str] = []
    for concept in graph.list_concepts():
        names = (concept.name, *concept.aliases)
        if any(_phrase_matches(text_tokens, name) for name in names):
            inferred.append(concept.id)
    return tuple(sorted(set(inferred)))


def _term_score(query_terms: frozenset[str], concept: Concept) -> float:
    target_terms = set(tokenize(concept.name)) | set(tokenize(concept.description))
    for alias in concept.aliases:
        target_terms |= set(tokenize(alias))
    if not query_terms or not target_terms:
        return 0.0
    return len(query_terms & target_terms) / len(query_terms | target_terms)


def _technique_paths(
    graph: InMemoryGraphRepository,
    query_concepts: tuple[str, ...],
    target_id: str,
) -> tuple[EvidencePath, ...]:
    paths: list[EvidencePath] = []
    for concept_id in query_concepts:
        if concept_id == target_id:
            paths.append(
                EvidencePath(
                    nodes=("query", target_id),
                    relations=("MENTIONS",),
                    score=0.6,
                    rationale="The query explicitly mentions this technique.",
                )
            )
            continue
        paths.extend(
            graph.find_paths(
                concept_id,
                target_id,
                max_hops=2,
                excluded_relation_types=("EXCLUDES",),
            )
        )
    return tuple(sorted(paths, key=lambda item: item.score, reverse=True))


def _fit_signals(
    graph: InMemoryGraphRepository,
    query_concepts: tuple[str, ...],
    target_id: str,
) -> tuple[str, ...]:
    signals: list[str] = []
    for concept_id in query_concepts:
        concept = graph.get_concept(concept_id)
        if concept and concept_id != target_id:
            signals.append(concept.name)
    return tuple(signals[:4])


def _pitfalls(concept: Concept) -> tuple[str, ...]:
    pitfall = concept.metadata.get("pitfall")
    return (pitfall,) if pitfall else ()


def _weighted_score(
    mode: str,
    *,
    vector_score: float,
    graph_score: float,
    concept_score: float,
) -> float:
    normalized_mode = mode.lower()
    if normalized_mode == "vector":
        return vector_score
    if normalized_mode == "graph":
        return graph_score
    return 0.35 * vector_score + 0.45 * graph_score + 0.20 * concept_score


def _confidence(score: float) -> str:
    if score >= 0.65:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def _phrase_matches(text_tokens: frozenset[str], phrase: str) -> bool:
    phrase_tokens = tokenize(phrase)
    return bool(phrase_tokens) and phrase_tokens.issubset(text_tokens)

import pytest

from backend.app.domain import Concept, Problem
from backend.app.repositories import InMemoryGraphRepository, InMemoryVectorRepository
from backend.app.services.retrieval import HybridRetrievalService


def build_service():
    graph = InMemoryGraphRepository()
    vectors = InMemoryVectorRepository()

    graph.add_concept(Concept(id="c_graph", name="graph"))
    graph.add_concept(Concept(id="c_dijkstra", name="dijkstra", aliases=("shortest path",)))
    graph.add_concept(Concept(id="c_greedy", name="greedy"))

    query = Problem(
        id="p_query",
        title="Network delay",
        text="Find shortest path in a weighted graph with non-negative edges.",
        concept_ids=("c_graph", "c_dijkstra"),
    )
    best = Problem(
        id="p_dijkstra",
        title="Dijkstra practice",
        text="Use dijkstra for shortest path on a weighted graph.",
        concept_ids=("c_graph", "c_dijkstra"),
    )
    vector_only = Problem(
        id="p_vector_only",
        title="Graph traversal",
        text="Find paths in a weighted graph with graph traversal.",
        concept_ids=("c_graph",),
    )
    excluded = Problem(
        id="p_excluded",
        title="Negative cycle",
        text="Shortest path in a weighted graph with negative edges.",
        concept_ids=("c_graph",),
    )

    for problem in (query, best, vector_only, excluded):
        graph.add_problem(problem)

    for problem in (best, vector_only, excluded):
        vectors.upsert_problem(problem)

    graph.add_relationship("p_query", "p_dijkstra", "SIMILAR_TO", weight=1.0)
    graph.add_relationship("p_query", "p_excluded", "EXCLUDES", weight=1.0)
    graph.add_relationship("p_dijkstra", "c_greedy", "EXCLUDES", weight=1.0)

    return HybridRetrievalService(graph, vectors), graph, vectors


def test_hybrid_ranking_prefers_graph_and_concept_evidence_over_vector_only():
    service, _, _ = build_service()

    bundle = service.recommend(problem_id="p_query", top_k=3)

    assert [item.problem_id for item in bundle.recommendations][:2] == [
        "p_dijkstra",
        "p_vector_only",
    ]
    best = bundle.recommendations[0]
    assert best.graph_score > 0
    assert best.concept_match_score > 0
    assert best.score > bundle.recommendations[1].score


def test_excludes_relationship_removes_blocked_candidates():
    service, _, _ = build_service()

    bundle = service.recommend(problem_id="p_query", top_k=5)

    assert "p_excluded" not in [item.problem_id for item in bundle.recommendations]


def test_recommendation_contains_inspectable_evidence_path():
    service, _, _ = build_service()

    bundle = service.recommend(problem_id="p_query", top_k=1)

    recommendation = bundle.recommendations[0]
    assert recommendation.problem_id == "p_dijkstra"
    assert recommendation.evidence_paths
    assert recommendation.evidence_paths[0].nodes == ("p_query", "p_dijkstra")
    assert recommendation.evidence_paths[0].relations == ("SIMILAR_TO",)


def test_text_query_can_retrieve_without_persisted_problem():
    service, _, _ = build_service()

    bundle = service.recommend(
        problem_text="Need shortest path in a non-negative weighted graph.",
        top_k=1,
    )

    assert bundle.query_problem.id == "query"
    assert bundle.recommendations[0].problem_id == "p_dijkstra"


def test_graph_only_candidate_can_be_recalled_without_vector_overlap():
    graph = InMemoryGraphRepository()
    vectors = InMemoryVectorRepository()
    graph.add_concept(Concept(id="c_graph", name="graph"))

    query = Problem(
        id="p_query",
        title="Query",
        text="graph",
        concept_ids=("c_graph",),
    )
    graph_only = Problem(
        id="p_graph_only",
        title="Graph-only evidence",
        text="zzzz yyyy xxxx",
        concept_ids=(),
    )
    graph.add_problem(query)
    graph.add_problem(graph_only)
    vectors.upsert_problem(graph_only)
    graph.add_relationship("p_query", "p_graph_only", "SIMILAR_TO", weight=1.0)

    bundle = HybridRetrievalService(graph, vectors).recommend(problem_id="p_query", top_k=1)

    assert bundle.recommendations[0].problem_id == "p_graph_only"
    assert bundle.recommendations[0].vector_score == 0
    assert bundle.recommendations[0].graph_score > 0


def test_recommend_rejects_invalid_top_k():
    service, _, _ = build_service()

    with pytest.raises(ValueError, match="top_k"):
        service.recommend(problem_id="p_query", top_k=0)


def test_concept_inference_uses_token_boundaries():
    graph = InMemoryGraphRepository()
    vectors = InMemoryVectorRepository()
    graph.add_concept(Concept(id="c_graph", name="graph"))
    candidate = Problem(
        id="p_candidate",
        title="Graph candidate",
        text="graph traversal",
        concept_ids=("c_graph",),
    )
    graph.add_problem(candidate)
    vectors.upsert_problem(candidate)

    bundle = HybridRetrievalService(graph, vectors).recommend(
        problem_text="Analyze a paragraph of text.",
        top_k=1,
    )

    assert bundle.query_problem.concept_ids == ()

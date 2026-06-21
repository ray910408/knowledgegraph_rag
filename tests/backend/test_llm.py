from backend.app.domain import EvidenceBundle, EvidencePath, Problem, Recommendation
from backend.app.services.llm import MockLLMProvider


def test_mock_llm_generates_only_from_evidence_without_api_key():
    provider = MockLLMProvider()
    bundle = EvidenceBundle(
        query_problem=Problem(
            id="p_query",
            title="Network delay",
            text="Find shortest path in a graph.",
            concept_ids=("c_graph",),
        ),
        recommendations=(
            Recommendation(
                problem_id="p_dijkstra",
                title="Dijkstra practice",
                score=0.8,
                vector_score=0.6,
                graph_score=1.0,
                concept_match_score=0.5,
                concept_matches=("graph",),
                evidence_paths=(
                    EvidencePath(
                        nodes=("p_query", "p_dijkstra"),
                        relations=("SIMILAR_TO",),
                        score=1.0,
                        rationale="direct graph evidence",
                    ),
                ),
            ),
        ),
    )

    explanation = provider.explain(bundle)

    assert "Dijkstra practice" in explanation
    assert "SIMILAR_TO" in explanation
    assert "graph" in explanation
    assert "source code" not in explanation.lower()

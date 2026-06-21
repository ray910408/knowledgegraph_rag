from __future__ import annotations

from backend.app.providers import DeterministicMockEmbeddingProvider
from backend.app.retrieval.pipeline import (
    BM25SearchService,
    ContextBuilder,
    EntityLinkingService,
    EvidenceBuilder,
    GraphSearchService,
    HybridFusionService,
    OnlineQueryPipeline,
    QueryUnderstandingService,
    Reranker,
    RetrievalCandidate,
    RetrievalDocument,
    VectorSearchService,
)


def _documents() -> tuple[RetrievalDocument, ...]:
    return (
        RetrievalDocument(
            id="leetcode-994",
            source="LeetCode",
            source_id="994",
            title="Rotting Oranges",
            text="Multi-source BFS with a queue on a grid.",
            answer="Use BFS from all rotten oranges.",
            concepts=("BFS", "Queue"),
            problem_type="Graph Traversal",
        ),
        RetrievalDocument(
            id="leetcode-300",
            source="LeetCode",
            source_id="300",
            title="Longest Increasing Subsequence",
            text="Dynamic programming over increasing subsequences.",
            answer="Use DP.",
            concepts=("Dynamic Programming",),
            problem_type="Dynamic Programming",
        ),
    )


def test_vector_graph_and_bm25_search_return_candidates():
    documents = _documents()
    embedding_provider = DeterministicMockEmbeddingProvider(dimension=8)
    understanding = QueryUnderstandingService().understand("BFS shortest path with queue")
    linked_entities = EntityLinkingService().link(understanding)

    vector_candidates = VectorSearchService(documents, embedding_provider).search(
        understanding,
        top_k=2,
    )
    bm25_candidates = BM25SearchService(documents).search(understanding, top_k=2)
    graph_result = GraphSearchService(documents).search(linked_entities, top_k=2)

    assert vector_candidates
    assert all(candidate.source == "vector" for candidate in vector_candidates)
    assert bm25_candidates[0].id == "leetcode-994"
    assert bm25_candidates[0].source == "bm25"
    assert graph_result.candidates[0].id == "leetcode-994"
    assert graph_result.paths[0]["nodes"] == ["input", "concept:bfs", "leetcode-994"]


def test_hybrid_fusion_dedupes_and_normalizes_scores_then_reranks():
    candidates = HybridFusionService().fuse(
        vector_candidates=(
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="vector",
                score=0.4,
                text="BFS queue",
                concepts=("BFS", "Queue"),
            ),
        ),
        graph_candidates=(
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="graph",
                score=1.0,
                text="BFS queue",
                concepts=("BFS", "Queue"),
            ),
        ),
        bm25_candidates=(
            RetrievalCandidate(
                id="leetcode-300",
                title="Longest Increasing Subsequence",
                source="bm25",
                score=3.0,
                text="dynamic programming",
                concepts=("Dynamic Programming",),
            ),
        ),
        top_k=3,
    )

    assert [candidate.id for candidate in candidates] == ["leetcode-994", "leetcode-300"]
    assert candidates[0].score <= 1.0
    assert candidates[0].payload["sources"] == ["graph", "vector"]

    reranked = Reranker().rerank("BFS queue", candidates, top_k=2)
    assert reranked[0].id == "leetcode-994"
    assert reranked[0].payload["rerankerScore"] >= reranked[1].payload["rerankerScore"]


def test_evidence_and_context_builders_create_stable_llm_context():
    documents = _documents()
    result = OnlineQueryPipeline(documents=documents).run(
        "BFS shortest path with queue",
        top_k=2,
    )

    evidence = EvidenceBuilder().build(result.reranked_candidates, result.graph_paths)
    context = ContextBuilder().build(result.query_understanding, evidence)

    evidence_map = evidence.to_mapping()
    assert evidence_map["similarProblems"][0]["id"] == "leetcode-994"
    assert evidence_map["graphPaths"]
    assert "BFS" in evidence_map["algorithmEvidence"]
    assert "Queue" in evidence_map["dataStructureEvidence"]
    assert "Graph Traversal" in evidence_map["patternEvidence"]
    assert "Query Understanding" in context
    assert "Rotting Oranges" in context
    assert "Common Mistakes" in context


def test_online_pipeline_trace_has_required_debug_sections():
    result = OnlineQueryPipeline(documents=_documents()).run("BFS shortest path", top_k=2)

    trace = result.trace.to_mapping()
    assert trace["queryUnderstanding"]["intent"] == "problem_search"
    assert trace["entityLinking"]
    assert trace["vectorCandidates"]
    assert trace["graphCandidates"]
    assert trace["bm25Candidates"]
    assert trace["fusionScores"]
    assert trace["rerankerScores"]

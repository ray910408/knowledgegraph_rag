from __future__ import annotations

from backend.app.adapters.in_memory import (
    InMemoryBM25Store,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from backend.app.contracts import EntityRecord, RelationRecord
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
from backend.app.stores import BM25Document, VectorRecord


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


def _store_payload(document: RetrievalDocument) -> dict[str, object]:
    return {
        "problemId": document.id,
        "kind": "statement",
        "text": document.text,
        "concepts": list(document.concepts),
        "metadata": {
            "source": document.source,
            "sourceId": document.source_id,
            "title": document.title,
            "problemType": document.problem_type,
            "answer": document.answer,
        },
    }


def _build_vector_store(
    documents: tuple[RetrievalDocument, ...],
    embedding_provider: DeterministicMockEmbeddingProvider,
) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    records = []
    for document in documents:
        text = (
            "BFS shortest path with queue"
            if document.id == "leetcode-994"
            else f"{document.title} {document.text}"
        )
        records.append(
            VectorRecord(
                id=f"{document.id}:statement:0",
                vector=tuple(embedding_provider.embed_text(text)),
                payload=_store_payload(document),
            )
        )
    store.upsert(tuple(records))
    return store


def _build_bm25_store(documents: tuple[RetrievalDocument, ...]) -> InMemoryBM25Store:
    store = InMemoryBM25Store()
    store.index_documents(
        tuple(
            BM25Document(
                id=f"{document.id}:statement:0",
                text=f"{document.title} {document.text} {document.answer}",
                payload=_store_payload(document),
            )
            for document in documents
        )
    )
    return store


def _build_graph_store(documents: tuple[RetrievalDocument, ...]) -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.upsert_entities(
        (
            EntityRecord(id="concept:bfs", name="BFS", type="algorithm"),
            EntityRecord(id="concept:queue", name="Queue", type="data_structure"),
            EntityRecord(id="pattern:graph-traversal", name="Graph Traversal", type="pattern"),
            *(
                EntityRecord(
                    id=document.id,
                    name=document.title,
                    type="problem",
                    metadata={
                        "source": document.source,
                        "sourceId": document.source_id,
                        "problemType": document.problem_type,
                    },
                )
                for document in documents
            ),
        )
    )
    store.upsert_relations(
        (
            RelationRecord(
                id="leetcode-994->concept:bfs",
                source_id="leetcode-994",
                target_id="concept:bfs",
                type="REQUIRES",
                weight=1.0,
            ),
            RelationRecord(
                id="leetcode-994->concept:queue",
                source_id="leetcode-994",
                target_id="concept:queue",
                type="REQUIRES",
                weight=1.0,
            ),
            RelationRecord(
                id="leetcode-994->pattern:graph-traversal",
                source_id="leetcode-994",
                target_id="pattern:graph-traversal",
                type="HAS_PATTERN",
                weight=1.0,
            ),
        )
    )
    return store


def test_vector_search_service_can_use_vector_store():
    documents = _documents()
    embedding_provider = DeterministicMockEmbeddingProvider(dimension=8)
    vector_store = _build_vector_store(documents, embedding_provider)
    understanding = QueryUnderstandingService().understand("BFS shortest path with queue")

    candidates = VectorSearchService(
        documents,
        embedding_provider,
        vector_store=vector_store,
    ).search(understanding, top_k=2)

    assert candidates[0].id == "leetcode-994"
    assert candidates[0].source == "vector"
    assert candidates[0].payload["storeCandidateId"] == "leetcode-994:statement:0"
    assert candidates[0].payload["documentSource"] == "LeetCode"


def test_bm25_search_service_can_use_bm25_store():
    documents = _documents()
    bm25_store = _build_bm25_store(documents)
    understanding = QueryUnderstandingService().understand("BFS queue shortest path")

    candidates = BM25SearchService(documents, bm25_store=bm25_store).search(
        understanding,
        top_k=2,
    )

    assert candidates[0].id == "leetcode-994"
    assert candidates[0].source == "bm25"
    assert candidates[0].payload["storeCandidateId"] == "leetcode-994:statement:0"
    assert candidates[0].payload["answer"] == "Use BFS from all rotten oranges."


def test_graph_search_service_can_use_graph_store():
    documents = _documents()
    graph_store = _build_graph_store(documents)
    understanding = QueryUnderstandingService().understand("BFS queue graph traversal")
    linked_entities = EntityLinkingService().link(understanding)

    result = GraphSearchService(documents, graph_store=graph_store).search(
        linked_entities,
        top_k=2,
    )

    assert result.candidates[0].id == "leetcode-994"
    assert result.candidates[0].source == "graph"
    assert any(
        path["nodes"] == ["input", "concept:bfs", "leetcode-994"]
        and path["relations"] == ["MENTIONS", "REQUIRED_BY"]
        and path["storePath"]["nodes"] == ["leetcode-994", "concept:bfs"]
        and path["storePath"]["relations"] == ["REQUIRES"]
        for path in result.paths
    )


def test_graph_search_service_scores_partial_store_entity_matches_by_coverage():
    documents = _documents()
    graph_store = _build_graph_store(documents)
    linked_entities = (
        {"entityId": "concept:bfs", "name": "BFS"},
        {"entityId": "concept:missing", "name": "Missing"},
    )

    result = GraphSearchService(documents, graph_store=graph_store).search(
        linked_entities,
        top_k=2,
    )

    assert result.candidates[0].id == "leetcode-994"
    assert result.candidates[0].score == round(1.0 / len(linked_entities), 6)
    assert any("storePath" in path for path in result.paths)


def test_online_pipeline_accepts_store_injection_and_preserves_debug_outputs():
    documents = _documents()
    embedding_provider = DeterministicMockEmbeddingProvider(dimension=8)
    pipeline = OnlineQueryPipeline(
        documents=documents,
        embedding_provider=embedding_provider,
        vector_store=_build_vector_store(documents, embedding_provider),
        bm25_store=_build_bm25_store(documents),
        graph_store=_build_graph_store(documents),
    )

    result = pipeline.run("BFS shortest path with queue", top_k=2)
    evidence = EvidenceBuilder().build(result.reranked_candidates, result.graph_paths)
    context = ContextBuilder().build(result.query_understanding, evidence)
    trace = result.trace.to_mapping()

    assert result.query_understanding.intent == "problem_search"
    assert result.vector_candidates[0].id == "leetcode-994"
    assert result.bm25_candidates[0].id == "leetcode-994"
    assert result.graph_candidates[0].id == "leetcode-994"
    assert trace["vectorCandidates"][0]["payload"]["storeCandidateId"]
    assert trace["bm25Candidates"][0]["payload"]["storeCandidateId"]
    assert trace["graphCandidates"]
    assert any("storePath" in path for path in result.graph_paths)
    assert evidence.to_mapping()["similarProblems"]
    assert "Query Understanding" in context


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


def test_hybrid_fusion_counts_each_source_once_per_problem():
    candidates = HybridFusionService().fuse(
        vector_candidates=(
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="vector",
                score=0.9,
                text="BFS queue chunk one",
                concepts=("BFS", "Queue"),
            ),
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="vector",
                score=0.8,
                text="BFS queue chunk two",
                concepts=("BFS", "Queue"),
            ),
            RetrievalCandidate(
                id="leetcode-300",
                title="Longest Increasing Subsequence",
                source="vector",
                score=0.6,
                text="dynamic programming",
                concepts=("Dynamic Programming",),
            ),
        ),
        graph_candidates=(),
        bm25_candidates=(),
        top_k=2,
    )

    by_id = {candidate.id: candidate for candidate in candidates}

    assert by_id["leetcode-994"].score == 0.35
    assert by_id["leetcode-994"].payload["sources"] == ["vector"]
    assert by_id["leetcode-300"].score == round(0.35 * (0.6 / 0.9), 6)


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

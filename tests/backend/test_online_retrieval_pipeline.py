from __future__ import annotations

import pytest

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
    ExactProblemMatcher,
    GraphSearchService,
    HybridFusionService,
    OnlineQueryPipeline,
    QueryUnderstandingService,
    Reranker,
    RetrievalCandidate,
    RetrievalDocument,
    VectorSearchService,
    _aggregate_problem_candidates,
)
from backend.app.stores import BM25Document, SearchCandidate, VectorRecord


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
            solution_hints=("Push all rotten oranges first.", "Expand one BFS layer per minute."),
            difficulty="Medium",
            constraints=("1 <= m, n <= 10",),
            examples=({"input": "grid", "output": "4"},),
            editorial="Use multi-source BFS from all rotten cells.",
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
        "answer": document.answer,
        "solutionHints": list(document.solution_hints),
        "difficulty": document.difficulty,
        "constraints": list(document.constraints),
        "examples": [dict(example) for example in document.examples],
        "editorial": document.editorial,
        "source": document.source,
        "sourceId": document.source_id,
        "title": document.title,
        "problemType": document.problem_type,
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


class _FakeVectorStore:
    def __init__(self, candidates: tuple[SearchCandidate, ...]) -> None:
        self._candidates = candidates
        self.requested_top_k: list[int] = []

    def search(self, query_vector, *, top_k, filters=None):
        self.requested_top_k.append(top_k)
        return self._candidates[:top_k]


class _FakeBM25Store:
    def __init__(self, candidates: tuple[SearchCandidate, ...]) -> None:
        self._candidates = candidates
        self.requested_top_k: list[int] = []

    def search(self, query, *, top_k):
        self.requested_top_k.append(top_k)
        return self._candidates[:top_k]


def test_store_backed_bm25_filters_zero_score_candidates():
    uva = _uva_document()
    leetcode = _documents()[0]
    bm25_store = _FakeBM25Store(
        (
            SearchCandidate(
                id="leetcode-994:statement:0",
                score=0.0,
                payload=_store_payload(leetcode),
            ),
            SearchCandidate(
                id="uva-10653:statement:0",
                score=0.7,
                payload=_store_payload(uva),
            ),
        )
    )
    understanding = QueryUnderstandingService((leetcode, uva)).understand("10653")

    candidates = BM25SearchService((leetcode, uva), bm25_store=bm25_store).search(
        understanding,
        top_k=3,
    )

    assert [candidate.id for candidate in candidates] == ["uva-10653"]
    assert all(candidate.score > 0 for candidate in candidates)


def _duplicate_chunk_candidates() -> tuple[SearchCandidate, ...]:
    first, second = _documents()
    return (
        SearchCandidate(
            id="leetcode-994:statement:0",
            score=0.90,
            payload=_store_payload(first),
        ),
        SearchCandidate(
            id="leetcode-994:answer:1",
            score=0.80,
            payload={**_store_payload(first), "kind": "answer"},
        ),
        SearchCandidate(
            id="leetcode-994:hint-1:2",
            score=0.70,
            payload={**_store_payload(first), "kind": "hint"},
        ),
        SearchCandidate(
            id="leetcode-300:statement:0",
            score=0.60,
            payload=_store_payload(second),
        ),
    )


def _interleaved_chunk_candidates() -> tuple[SearchCandidate, ...]:
    first, second = _documents()
    return (
        SearchCandidate(
            id="leetcode-994:statement:0",
            score=0.90,
            payload=_store_payload(first),
        ),
        SearchCandidate(
            id="leetcode-300:statement:0",
            score=0.80,
            payload=_store_payload(second),
        ),
        SearchCandidate(
            id="leetcode-994:answer:1",
            score=0.70,
            payload={**_store_payload(first), "kind": "answer"},
        ),
    )


def _same_problem_candidates(count: int) -> tuple[SearchCandidate, ...]:
    first, _ = _documents()
    return tuple(
        SearchCandidate(
            id=f"leetcode-994:chunk:{index}",
            score=1.0 - (index / 1000),
            payload={**_store_payload(first), "kind": "chunk"},
        )
        for index in range(count)
    )


def test_store_enriches_once_after_early_unique_success():
    documents = _documents()
    store = _FakeVectorStore(_interleaved_chunk_candidates())
    understanding = QueryUnderstandingService().understand("BFS queue")

    candidates = VectorSearchService(
        documents,
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=store,
    ).search(understanding, top_k=2)

    assert store.requested_top_k == [2, 4]
    assert [
        chunk["payload"]["storeCandidateId"]
        for chunk in candidates[0].payload["rawChunks"]
    ] == [
        "leetcode-994:statement:0",
        "leetcode-994:answer:1",
    ]


def test_store_fetch_attempts_and_windows_are_tightly_bounded():
    documents = _documents()
    understanding = QueryUnderstandingService().understand("BFS queue")
    attempt_store = _FakeVectorStore(_same_problem_candidates(120))
    cap_store = _FakeVectorStore(_same_problem_candidates(120))

    VectorSearchService(
        documents,
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=attempt_store,
    ).search(understanding, top_k=10)
    capped_candidates = VectorSearchService(
        documents,
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=cap_store,
    ).search(understanding, top_k=30)

    assert attempt_store.requested_top_k == [10, 20, 40, 80]
    assert cap_store.requested_top_k == [30, 60, 100]
    assert capped_candidates[0].payload["rawChunksComplete"] is False


def test_store_marks_raw_chunks_complete_only_after_a_short_page():
    documents = _documents()
    understanding = QueryUnderstandingService().understand("BFS queue")
    exhausted_store = _FakeVectorStore(_interleaved_chunk_candidates())
    capped_store = _FakeVectorStore(_same_problem_candidates(120))

    exhausted = VectorSearchService(
        documents,
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=exhausted_store,
    ).search(understanding, top_k=2)
    capped = VectorSearchService(
        documents,
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=capped_store,
    ).search(understanding, top_k=30)

    assert exhausted[0].payload["rawChunksComplete"] is True
    assert capped[0].payload["rawChunksComplete"] is False


def test_aggregate_problem_candidates_deep_copies_nested_chunk_snapshots():
    nested_payload = {
        "storeCandidateId": "uva-10653:answer:1",
        "storePayload": {"metadata": {"tags": ["original"]}},
    }
    chunk = RetrievalCandidate(
        id="uva-10653",
        title="Bombs! NO they are Mines!!",
        source="vector",
        score=0.40,
        payload=nested_payload,
    )

    aggregated = _aggregate_problem_candidates((chunk,), source="vector", top_k=1)
    aggregate_payload = aggregated[0].payload
    raw_chunk_payload = aggregate_payload["rawChunks"][0]["payload"]

    nested_payload["storePayload"]["metadata"]["tags"].append("mutated")
    assert aggregate_payload["storePayload"]["metadata"]["tags"] == ["original"]
    assert raw_chunk_payload["storePayload"]["metadata"]["tags"] == ["original"]

    aggregate_payload["storePayload"]["metadata"]["tags"].append("aggregate-only")
    assert raw_chunk_payload["storePayload"]["metadata"]["tags"] == ["original"]


def test_vector_store_overfetches_until_it_has_requested_unique_problems():
    documents = _documents()
    store = _FakeVectorStore(_duplicate_chunk_candidates())
    understanding = QueryUnderstandingService().understand("BFS queue")

    candidates = VectorSearchService(
        documents,
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=store,
    ).search(understanding, top_k=2)

    assert store.requested_top_k == [2, 4, 8]
    assert [candidate.id for candidate in candidates] == ["leetcode-994", "leetcode-300"]
    assert [
        chunk["payload"]["storeCandidateId"]
        for chunk in candidates[0].payload["rawChunks"]
    ] == [
        "leetcode-994:statement:0",
        "leetcode-994:answer:1",
        "leetcode-994:hint-1:2",
    ]


def test_bm25_store_overfetches_until_it_has_requested_unique_problems():
    documents = _documents()
    store = _FakeBM25Store(_duplicate_chunk_candidates())
    understanding = QueryUnderstandingService().understand("BFS queue")

    candidates = BM25SearchService(documents, bm25_store=store).search(
        understanding,
        top_k=2,
    )

    assert store.requested_top_k == [2, 4, 8]
    assert [candidate.id for candidate in candidates] == ["leetcode-994", "leetcode-300"]
    assert [
        chunk["payload"]["storeCandidateId"]
        for chunk in candidates[0].payload["rawChunks"]
    ] == [
        "leetcode-994:statement:0",
        "leetcode-994:answer:1",
        "leetcode-994:hint-1:2",
    ]


def test_vector_store_omits_non_positive_candidates():
    first, second = _documents()
    store = _FakeVectorStore(
        (
            SearchCandidate(
                id="leetcode-994:statement:0",
                score=0.50,
                payload=_store_payload(first),
            ),
            SearchCandidate(
                id="leetcode-300:statement:0",
                score=-0.25,
                payload=_store_payload(second),
            ),
        )
    )
    understanding = QueryUnderstandingService().understand("BFS queue")

    candidates = VectorSearchService(
        (first, second),
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=store,
    ).search(understanding, top_k=2)

    assert [candidate.id for candidate in candidates] == ["leetcode-994"]


def test_store_candidate_payload_preserves_enriched_evidence_fields():
    documents = _documents()
    embedding_provider = DeterministicMockEmbeddingProvider(dimension=8)
    vector_store = _build_vector_store(documents, embedding_provider)
    bm25_store = _build_bm25_store(documents)
    understanding = QueryUnderstandingService().understand("BFS shortest path with queue")

    vector_candidate = VectorSearchService(
        documents,
        embedding_provider,
        vector_store=vector_store,
    ).search(understanding, top_k=1)[0]
    bm25_candidate = BM25SearchService(documents, bm25_store=bm25_store).search(
        understanding,
        top_k=1,
    )[0]

    for candidate in (vector_candidate, bm25_candidate):
        assert candidate.payload["answer"] == "Use BFS from all rotten oranges."
        assert candidate.payload["solutionHints"] == [
            "Push all rotten oranges first.",
            "Expand one BFS layer per minute.",
        ]
        assert candidate.payload["difficulty"] == "Medium"
        assert candidate.payload["constraints"] == ["1 <= m, n <= 10"]
        assert candidate.payload["sourceId"] == "994"
        assert candidate.payload["title"] == "Rotting Oranges"
        assert candidate.payload["problemType"] == "Graph Traversal"
        assert candidate.payload["concepts"] == ["BFS", "Queue"]


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
        and path["pathSource"] == "neo4j"
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
    assert any(
        path["pathSource"] == "neo4j" and "storePath" in path
        for path in result.paths
    )


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
    assert trace["vectorCandidates"][0]["payload"]["chunkCount"] == 1
    assert trace["bm25Candidates"][0]["payload"]["chunkCount"] == 1
    assert trace["vectorCandidates"][0]["payload"]["rawChunks"]
    assert trace["bm25Candidates"][0]["payload"]["rawChunks"]
    assert trace["graphCandidates"]
    assert any(
        path["pathSource"] == "neo4j" and "storePath" in path
        for path in result.graph_paths
    )
    assert evidence.to_mapping()["similarProblems"]
    assert "查詢理解" in context


@pytest.mark.parametrize(
    ("mode", "expected_ids", "expected_sources"),
    [
        (
            "hybrid",
            {"vector-only", "graph-only", "bm25-only"},
            {
                "vector-only": ["vector"],
                "graph-only": ["graph"],
                "bm25-only": ["bm25"],
            },
        ),
        ("vector", {"vector-only"}, {"vector-only": ["vector"]}),
        ("graph", {"graph-only"}, {"graph-only": ["graph"]}),
    ],
)
def test_online_pipeline_mode_controls_fusion_sources_and_final_results(
    mode,
    expected_ids,
    expected_sources,
):
    documents = (
        RetrievalDocument(
            id="vector-only",
            source="Test",
            source_id="vector",
            title="Vector candidate",
            text="Embedding-only candidate.",
            answer="Vector answer.",
            concepts=("Embeddings",),
            problem_type="Similarity",
        ),
        RetrievalDocument(
            id="graph-only",
            source="Test",
            source_id="graph",
            title="Graph candidate",
            text="BFS graph candidate.",
            answer="Graph answer.",
            concepts=("BFS",),
            problem_type="Graph Traversal",
        ),
        RetrievalDocument(
            id="bm25-only",
            source="Test",
            source_id="bm25",
            title="BM25 candidate",
            text="Lexical-only candidate.",
            answer="BM25 answer.",
            concepts=("Lexical Search",),
            problem_type="Search",
        ),
    )
    vector_store = _FakeVectorStore(
        (
            SearchCandidate(
                id="vector-only:statement:0",
                score=0.9,
                payload=_store_payload(documents[0]),
            ),
        )
    )
    bm25_store = _FakeBM25Store(
        (
            SearchCandidate(
                id="bm25-only:statement:0",
                score=0.8,
                payload=_store_payload(documents[2]),
            ),
        )
    )

    result = OnlineQueryPipeline(
        documents=documents,
        vector_store=vector_store,
        bm25_store=bm25_store,
    ).run("BFS", mode=mode, top_k=3)
    trace = result.trace.to_mapping()

    assert {candidate.id for candidate in result.fused_candidates} == expected_ids
    assert {candidate.id for candidate in result.reranked_candidates} == expected_ids
    assert {
        candidate.id: candidate.payload["sources"]
        for candidate in result.fused_candidates
    } == expected_sources
    assert trace["vectorCandidates"]
    assert trace["graphCandidates"]
    assert trace["bm25Candidates"]
    assert {candidate["id"] for candidate in trace["fusionScores"]} == expected_ids
    assert {candidate["id"] for candidate in trace["rerankerScores"]} == expected_ids
    assert result.graph_paths


def test_online_pipeline_top_k_limits_mode_specific_final_results():
    result = OnlineQueryPipeline(documents=_documents()).run(
        "BFS shortest path with queue",
        mode="vector",
        top_k=1,
    )

    assert len(result.fused_candidates) <= 2
    assert len(result.reranked_candidates) == 1
    assert len(EvidenceBuilder().build(result.reranked_candidates, ()).similar_problems) == 1


def test_online_pipeline_mode_keeps_exact_match_and_graph_paths_independent():
    similar_document = RetrievalDocument(
        id="leetcode-1091",
        source="LeetCode",
        source_id="1091",
        title="Shortest Path in Binary Matrix",
        text="Use BFS to find a shortest path in an unweighted binary matrix.",
        answer="Run BFS over eight directions.",
        concepts=("BFS", "Queue", "Visited Array"),
        problem_type="Graph Traversal",
    )
    result = OnlineQueryPipeline(
        documents=(_uva_document(), similar_document),
        vector_store=_FakeVectorStore(
            (
                SearchCandidate(
                    id="leetcode-1091:statement:0",
                    score=0.9,
                    payload=_store_payload(similar_document),
                ),
            )
        ),
        bm25_store=_FakeBM25Store(()),
    ).run(
        "UVA-10653 - Bombs! NO they are Mines!!",
        mode="vector",
        top_k=1,
    )

    assert result.matched_problem is not None
    assert result.matched_problem.problem_id == "uva-10653"
    assert result.graph_paths
    assert [candidate.id for candidate in result.reranked_candidates] == ["leetcode-1091"]


def test_store_chunk_candidates_are_aggregated_by_problem_and_keep_raw_chunks():
    chunk_one = RetrievalCandidate(
        id="uva-10653",
        title="Bombs! NO they are Mines!!",
        source="vector",
        score=0.20,
        text="BFS queue",
        concepts=("BFS", "Queue"),
        problem_type="Graph Traversal",
        payload={"storeCandidateId": "uva-10653:answer:1", "kind": "answer"},
    )
    chunk_two = RetrievalCandidate(
        id="uva-10653",
        title="Bombs! NO they are Mines!!",
        source="vector",
        score=0.40,
        text="visited grid",
        concepts=("BFS", "Visited Array"),
        problem_type="Graph Traversal",
        payload={"storeCandidateId": "uva-10653:hint-1:3", "kind": "hint"},
    )
    other = RetrievalCandidate(
        id="leetcode-1091",
        title="Shortest Path in Binary Matrix",
        source="vector",
        score=0.30,
        text="shortest path",
        concepts=("BFS",),
        problem_type="Graph Traversal",
        payload={"storeCandidateId": "leetcode-1091:statement:0"},
    )

    aggregated = _aggregate_problem_candidates((chunk_one, chunk_two, other), source="vector", top_k=5)

    assert [candidate.id for candidate in aggregated] == ["uva-10653", "leetcode-1091"]
    assert aggregated[0].score == 0.40
    assert aggregated[0].payload["chunkCount"] == 2
    assert [chunk["payload"]["storeCandidateId"] for chunk in aggregated[0].payload["rawChunks"]] == [
        "uva-10653:hint-1:3",
        "uva-10653:answer:1",
    ]


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
    assert graph_result.paths[0]["pathSource"] == "inferred"


@pytest.mark.parametrize(
    "query",
    (
        "10653",
        "uva-10653",
        "uva 10653",
        "Bombs! NO they are Mines!!",
        "UVa Bombs! NO they are Mines!!",
    ),
)
def test_local_bm25_scores_exact_problem_aliases(query: str):
    documents = (_documents()[0], _uva_document())
    understanding = QueryUnderstandingService(documents).understand(query)

    candidates = BM25SearchService(documents).search(understanding, top_k=3)

    assert [candidate.id for candidate in candidates] == ["uva-10653"]
    assert candidates[0].score > 0
    assert candidates[0].source == "bm25"


def test_local_bm25_filters_zero_score_candidates():
    documents = (_documents()[0], _uva_document())
    understanding = QueryUnderstandingService(documents).understand("10653")

    candidates = BM25SearchService(documents).search(understanding, top_k=3)

    assert [candidate.id for candidate in candidates] == ["uva-10653"]
    assert all(candidate.score > 0 for candidate in candidates)


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


def test_hybrid_fusion_ignores_non_positive_candidates_from_every_source():
    candidates = HybridFusionService().fuse(
        vector_candidates=(
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="vector",
                score=0.8,
                text="BFS queue",
                concepts=("BFS", "Queue"),
            ),
            RetrievalCandidate(
                id="uva-10653",
                title="Bombs! NO they are Mines!!",
                source="vector",
                score=0.0,
                text="untrusted adapter row",
                concepts=("BFS",),
            ),
        ),
        graph_candidates=(
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="graph",
                score=-0.1,
                text="bad graph row",
                concepts=("BFS",),
            ),
        ),
        bm25_candidates=(
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="bm25",
                score=0.0,
                text="bad BM25 row",
                concepts=("BFS",),
            ),
        ),
        top_k=3,
    )

    assert [candidate.id for candidate in candidates] == ["leetcode-994"]
    assert candidates[0].score == 0.35
    assert candidates[0].payload["sources"] == ["vector"]


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
    assert "查詢理解" in context
    assert "Rotting Oranges" in context
    assert "常見錯誤" in context


def test_context_builder_includes_enriched_candidate_evidence():
    candidate = RetrievalCandidate(
        id="leetcode-994",
        title="Rotting Oranges",
        source="hybrid",
        score=0.97,
        text="Multi-source BFS with a queue on a grid.",
        concepts=("BFS", "Queue"),
        problem_type="Graph Traversal",
        payload={
            "answer": "Use BFS from all rotten oranges.",
            "solutionHints": ["Push all rotten oranges first."],
            "difficulty": "Medium",
            "constraints": ["1 <= m, n <= 10"],
            "documentSource": "LeetCode",
            "sourceId": "994",
        },
    )
    understanding = QueryUnderstandingService().understand("BFS queue shortest path")
    graph_paths = (
        {
            "nodes": ["input", 42, "leetcode-994"],
            "relations": ["MENTIONS", 7],
            "rationale": "linked BFS to Rotting Oranges",
        },
    )

    evidence = EvidenceBuilder().build((candidate,), graph_paths)
    context = ContextBuilder().build(understanding, evidence)

    similar_problem = evidence.to_mapping()["similarProblems"][0]
    assert similar_problem["answerHint"] == "Use BFS from all rotten oranges."
    assert similar_problem["solutionHints"] == ["Push all rotten oranges first."]
    assert similar_problem["difficulty"] == "Medium"
    assert similar_problem["constraints"] == ["1 <= m, n <= 10"]
    assert "查詢理解" in context
    assert "- 意圖: problem_search" in context
    assert "- 輸入類型: problem" in context
    assert "- 關鍵詞: bfs, queue, shortest, path" in context
    assert "命中題目\n- 無" in context
    assert "相似題" in context
    assert "答案摘要: Use BFS from all rotten oranges." in context
    assert "解題提示: Push all rotten oranges first." in context
    assert "難度: Medium" in context
    assert "限制: 1 <= m, n <= 10" in context
    assert "圖路徑" in context
    assert "input -> 42 -> leetcode-994" in context
    assert "relations=MENTIONS, 7" in context
    assert "rationale=linked BFS to Rotting Oranges" in context
    assert "演算法證據" in context
    assert "資料結構證據" in context
    assert "技巧證據" in context
    assert "題型證據" in context
    assert "常見錯誤" in context


def test_context_builder_includes_matched_problem_separately():
    matched = ExactProblemMatcher((_uva_document(),)).match(
        QueryUnderstandingService((_uva_document(),)).understand(
            "UVA-10653 - Bombs! NO they are Mines!!"
        )
    )
    assert matched is not None
    similar = RetrievalCandidate(
        id="leetcode-1091",
        title="Shortest Path in Binary Matrix",
        source="reranker",
        score=0.82,
        concepts=("BFS", "Queue"),
        problem_type="Graph Traversal",
        payload={"answer": "Run BFS over eight directions."},
    )
    evidence = EvidenceBuilder().build(
        (matched.candidate, similar),
        (),
        matched_problem=matched,
    )

    context = ContextBuilder().build(
        QueryUnderstandingService((_uva_document(),)).understand(
            "UVA-10653 - Bombs! NO they are Mines!!"
        ),
        evidence,
    )

    assert "命中題目" in context
    assert "id: uva-10653" in context
    assert "title: Bombs! NO they are Mines!!" in context
    assert "matchKind: exact_problem_id" in context
    assert "confidence: 1.0" in context
    assert "答案摘要: Run BFS from the start cell while skipping bomb cells." in context
    assert "解題提示: Mark bomb cells before BFS." in context
    assert "相似題" in context
    assert "leetcode-1091 Shortest Path in Binary Matrix" in context


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


def _uva_document() -> RetrievalDocument:
    return RetrievalDocument(
        id="uva-10653",
        source="UVa",
        source_id="10653",
        title="Bombs! NO they are Mines!!",
        text="Find the shortest safe path on a grid with bomb cells.",
        answer="Run BFS from the start cell while skipping bomb cells.",
        concepts=("BFS", "Queue", "Visited Array"),
        problem_type="Graph Traversal",
        solution_hints=("Mark bomb cells before BFS.", "Track visited grid cells when enqueued."),
        difficulty="Medium",
    )


def test_graph_search_for_exact_problem_returns_problem_node_paths_with_source_labels():
    document = _uva_document()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_entities(
        (
            EntityRecord(id=document.id, name=document.title, type="problem"),
            EntityRecord(id="concept:bfs", name="BFS", type="algorithm"),
            EntityRecord(id="concept:queue", name="Queue", type="data_structure"),
        )
    )
    graph_store.upsert_relations(
        (
            RelationRecord(
                id="uva-10653->concept:bfs",
                source_id=document.id,
                target_id="concept:bfs",
                type="REQUIRES",
                weight=1.0,
            ),
            RelationRecord(
                id="uva-10653->concept:queue",
                source_id=document.id,
                target_id="concept:queue",
                type="REQUIRES",
                weight=1.0,
            ),
        )
    )
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None
    linked_entities = EntityLinkingService().link(
        understanding,
        matched_problem=matched,
    )

    result = GraphSearchService((document,), graph_store=graph_store).search(
        linked_entities,
        matched_problem=matched,
        top_k=3,
    )

    assert result.candidates == ()
    assert result.paths
    paths_by_target = {path["nodes"][-1]: path for path in result.paths}
    assert paths_by_target["concept:bfs"]["pathSource"] == "neo4j"
    assert paths_by_target["concept:queue"]["pathSource"] == "neo4j"
    assert paths_by_target["concept:visited-array"]["pathSource"] == "inferred"
    assert paths_by_target["pattern:graph-traversal"]["pathSource"] == "inferred"
    assert ["input", "uva-10653", "concept:bfs"] in [
        path["nodes"] for path in result.paths
    ]


def test_graph_search_for_exact_problem_combines_partial_store_paths_with_inferred_missing_paths():
    document = _uva_document()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_entities(
        (
            EntityRecord(id=document.id, name=document.title, type="problem"),
            EntityRecord(id="concept:bfs", name="BFS", type="algorithm"),
        )
    )
    graph_store.upsert_relations(
        (
            RelationRecord(
                id="uva-10653->concept:bfs",
                source_id=document.id,
                target_id="concept:bfs",
                type="REQUIRES",
                weight=1.0,
            ),
        )
    )
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None

    result = GraphSearchService((document,), graph_store=graph_store).search(
        (),
        matched_problem=matched,
        top_k=3,
    )

    paths_by_target = {path["nodes"][-1]: path for path in result.paths}
    assert paths_by_target["concept:bfs"]["pathSource"] == "neo4j"
    assert paths_by_target["concept:queue"]["pathSource"] == "inferred"
    assert paths_by_target["concept:visited-array"]["pathSource"] == "inferred"
    assert paths_by_target["pattern:graph-traversal"]["pathSource"] == "inferred"
    assert paths_by_target["concept:bfs"]["storePath"]["nodes"] == [
        "uva-10653",
        "concept:bfs",
    ]


def test_graph_search_for_exact_problem_uses_reverse_store_path_with_canonical_public_path():
    document = _uva_document()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_entities(
        (
            EntityRecord(id=document.id, name=document.title, type="problem"),
            EntityRecord(id="concept:bfs", name="BFS", type="algorithm"),
        )
    )
    graph_store.upsert_relations(
        (
            RelationRecord(
                id="concept:bfs->uva-10653",
                source_id="concept:bfs",
                target_id=document.id,
                type="REQUIRED_BY",
                weight=1.0,
            ),
        )
    )
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None

    result = GraphSearchService((document,), graph_store=graph_store).search(
        (),
        matched_problem=matched,
        top_k=3,
    )

    bfs_paths = [path for path in result.paths if path["nodes"][-1] == "concept:bfs"]
    assert len(bfs_paths) == 1
    assert bfs_paths[0]["nodes"] == ["input", "uva-10653", "concept:bfs"]
    assert bfs_paths[0]["relations"] == ["EXACT_MATCH", "REQUIRED_BY"]
    assert bfs_paths[0]["pathSource"] == "neo4j"
    assert bfs_paths[0]["storePath"]["nodes"] == ["concept:bfs", "uva-10653"]
    assert bfs_paths[0]["storePath"]["relations"] == ["REQUIRED_BY"]


def test_graph_search_for_exact_problem_prefers_valid_reverse_store_path_over_malformed_direct_path():
    class MalformedDirectValidReverseGraphStore:
        def find_paths(self, source_id, target_id, *, max_hops=3):
            if (source_id, target_id) == ("uva-10653", "concept:bfs"):
                return (
                    {
                        "nodes": [],
                        "relations": [],
                        "score": "not-a-score",
                    },
                )
            if (source_id, target_id) == ("concept:bfs", "uva-10653"):
                return (
                    {
                        "nodes": ["concept:bfs", "uva-10653"],
                        "relations": ["REQUIRED_BY"],
                        "score": 1.0,
                    },
                )
            return ()

    document = _uva_document()
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None

    result = GraphSearchService(
        (document,),
        graph_store=MalformedDirectValidReverseGraphStore(),
    ).search(
        (),
        matched_problem=matched,
        top_k=3,
    )

    bfs_paths = [path for path in result.paths if path["nodes"][-1] == "concept:bfs"]
    assert len(bfs_paths) == 1
    assert bfs_paths[0]["nodes"] == ["input", "uva-10653", "concept:bfs"]
    assert bfs_paths[0]["relations"] == ["EXACT_MATCH", "REQUIRED_BY"]
    assert bfs_paths[0]["pathSource"] == "neo4j"
    assert bfs_paths[0]["storePath"]["nodes"] == ["concept:bfs", "uva-10653"]
    assert bfs_paths[0]["storePath"]["relations"] == ["REQUIRED_BY"]


@pytest.mark.parametrize("score", ["not-a-score", None, float("nan"), float("inf"), float("-inf")])
def test_graph_search_for_exact_problem_normalizes_malformed_store_paths_without_crashing(score):
    class MalformedGraphStore:
        def find_paths(self, source_id, target_id, *, max_hops=3):
            if target_id == "concept:bfs":
                return (
                    {
                        "nodes": "not-a-node-list",
                        "relations": {"bad": "relation-container"},
                        "score": score,
                    },
                )
            return ()

    document = _uva_document()
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None

    result = GraphSearchService((document,), graph_store=MalformedGraphStore()).search(
        (),
        matched_problem=matched,
        top_k=3,
    )

    bfs_path = next(path for path in result.paths if path["nodes"][-1] == "concept:bfs")
    assert bfs_path["score"] == 0.0
    assert bfs_path["pathSource"] == "neo4j"
    assert bfs_path["storePath"] == {"nodes": [], "relations": []}


def test_graph_search_for_exact_problem_deduplicates_direct_and_reverse_store_paths():
    class DuplicateDirectionGraphStore:
        def __init__(self) -> None:
            self.calls = []

        def find_paths(self, source_id, target_id, *, max_hops=3):
            self.calls.append((source_id, target_id, max_hops))
            if (source_id, target_id) == ("uva-10653", "concept:bfs"):
                return (
                    {
                        "nodes": ["uva-10653", "concept:bfs"],
                        "relations": ["REQUIRES"],
                        "score": 1.0,
                    },
                )
            if (source_id, target_id) == ("concept:bfs", "uva-10653"):
                return (
                    {
                        "nodes": ["concept:bfs", "uva-10653"],
                        "relations": ["REQUIRED_BY"],
                        "score": 1.0,
                    },
                )
            return ()

    document = _uva_document()
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None
    graph_store = DuplicateDirectionGraphStore()

    result = GraphSearchService(
        (document,),
        graph_store=graph_store,
    ).search(
        (),
        matched_problem=matched,
        top_k=3,
    )

    assert ("concept:bfs", "uva-10653", 3) in graph_store.calls
    assert [
        path["nodes"]
        for path in result.paths
        if path["nodes"] == ["input", "uva-10653", "concept:bfs"]
    ] == [["input", "uva-10653", "concept:bfs"]]


def test_graph_search_for_exact_problem_skips_empty_problem_type_store_target():
    class RecordingGraphStore:
        def __init__(self) -> None:
            self.find_paths_calls: list[tuple[str, str, int]] = []

        def find_paths(self, source_id, target_id, *, max_hops=3):
            self.find_paths_calls.append((source_id, target_id, max_hops))
            return ()

    document = RetrievalDocument(
        id="uva-10653",
        source="UVa",
        source_id="10653",
        title="Bombs! NO they are Mines!!",
        text="Find the shortest safe path on a grid with bomb cells.",
        answer="Run BFS from the start cell while skipping bomb cells.",
        concepts=("BFS",),
        problem_type="",
    )
    graph_store = RecordingGraphStore()
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None

    GraphSearchService((document,), graph_store=graph_store).search(
        (),
        matched_problem=matched,
        top_k=1,
    )

    assert graph_store.find_paths_calls == [
        ("uva-10653", "concept:bfs", 3),
        ("concept:bfs", "uva-10653", 3),
    ]
    assert all(
        target_id != "pattern:unknown"
        for _, target_id, _ in graph_store.find_paths_calls
    )


def test_graph_search_marks_document_concept_fallback_paths_as_inferred():
    document = _uva_document()
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None
    linked_entities = EntityLinkingService().link(
        understanding,
        matched_problem=matched,
    )

    result = GraphSearchService((document,)).search(
        linked_entities,
        matched_problem=matched,
        top_k=3,
    )

    assert result.candidates == ()
    assert result.paths
    assert all(path["pathSource"] == "inferred" for path in result.paths)
    assert all("storePath" not in path for path in result.paths)
    assert "not returned by Neo4j" in result.paths[0]["rationale"]


def test_exact_problem_matcher_recognizes_problem_id_source_id_and_title():
    matcher = ExactProblemMatcher((_uva_document(),))

    exact_id = matcher.match(QueryUnderstandingService().understand("UVA-10653"))
    exact_id_with_title = matcher.match(
        QueryUnderstandingService().understand("UVA-10653 - Bombs! NO they are Mines!!")
    )
    bare_source_id = matcher.match(QueryUnderstandingService().understand("10653"))
    exact_title = matcher.match(QueryUnderstandingService().understand("Bombs! NO they are Mines!!"))
    source_title_alias = matcher.match(QueryUnderstandingService().understand("UVa Bombs! NO they are Mines!!"))
    partial_title = matcher.match(QueryUnderstandingService().understand("Bombs mines shortest path"))

    assert exact_id is not None
    assert exact_id.problem_id == "uva-10653"
    assert exact_id.match_kind == "exact_problem_id"
    assert exact_id_with_title is not None
    assert exact_id_with_title.match_kind == "exact_problem_id"
    assert bare_source_id is not None
    assert bare_source_id.match_kind == "exact_source_id"
    assert exact_title is not None
    assert exact_title.match_kind == "exact_title"
    assert source_title_alias is not None
    assert source_title_alias.match_kind == "exact_title"
    assert partial_title is not None
    assert partial_title.match_kind == "partial_title"
    assert partial_title.confidence < exact_title.confidence


def test_partial_problem_match_does_not_override_python_input_kind():
    understanding = QueryUnderstandingService((_uva_document(),)).understand(
        "def solve():\n    bombs = mines = shortest = path = []"
    )

    assert understanding.input_kind == "python"


def test_online_pipeline_promotes_exact_problem_seed_without_polluting_similar_problems():
    documents = (
        _uva_document(),
        RetrievalDocument(
            id="leetcode-1091",
            source="LeetCode",
            source_id="1091",
            title="Shortest Path in Binary Matrix",
            text="Use BFS to find a shortest path in an unweighted binary matrix.",
            answer="Run BFS over eight directions.",
            concepts=("BFS", "Queue", "Visited Array"),
            problem_type="Graph Traversal",
        ),
    )

    result = OnlineQueryPipeline(
        documents=documents,
        vector_store=_FakeVectorStore(()),
        bm25_store=_FakeBM25Store(()),
    ).run("UVA-10653 - Bombs! NO they are Mines!!", top_k=2)
    evidence = EvidenceBuilder().build(
        result.reranked_candidates,
        result.graph_paths,
        matched_problem=result.matched_problem,
    )
    trace = result.trace.to_mapping()
    evidence_map = evidence.to_mapping()

    assert result.query_understanding.input_kind == "problem"
    assert result.matched_problem is not None
    assert result.matched_problem.problem_id == "uva-10653"
    assert trace["matchedProblem"]["id"] == "uva-10653"
    assert evidence_map["matchedProblem"]["id"] == "uva-10653"
    assert all(problem["id"] != "uva-10653" for problem in evidence_map["similarProblems"])
    assert all(candidate.id != "uva-10653" for candidate in result.graph_candidates)
    assert all(candidate.id != "uva-10653" for candidate in result.fused_candidates)
    assert all(candidate.id != "uva-10653" for candidate in result.reranked_candidates)
    assert result.graph_paths[0]["nodes"] == [
        "input",
        "uva-10653",
        "concept:bfs",
    ]
    assert result.graph_paths[0]["pathSource"] == "inferred"


def test_matched_problem_pin_does_not_change_similar_problem_ranking():
    matched = ExactProblemMatcher((_uva_document(),)).match(
        QueryUnderstandingService((_uva_document(),)).understand("UVA-10653 - Bombs! NO they are Mines!!")
    )
    assert matched is not None
    unrelated = RetrievalCandidate(
        id="leetcode-1091",
        title="Shortest Path in Binary Matrix",
        source="reranker",
        score=0.80,
        text="Use BFS with a queue over matrix states.",
        concepts=("BFS", "Queue"),
        problem_type="Graph Traversal",
    )
    weaker = RetrievalCandidate(
        id="leetcode-994",
        title="Rotting Oranges",
        source="reranker",
        score=0.60,
        text="Multi-source BFS with visited state tracking.",
        concepts=("BFS", "Queue", "State Tracking"),
        problem_type="Graph Traversal",
    )

    evidence = EvidenceBuilder().build(
        (unrelated, matched.candidate, weaker),
        (),
        matched_problem=matched,
    )
    evidence_map = evidence.to_mapping()

    assert evidence_map["matchedProblem"]["id"] == "uva-10653"
    assert [problem["id"] for problem in evidence_map["similarProblems"]] == [
        "leetcode-1091",
        "leetcode-994",
    ]
    assert "Queue" in evidence_map["dataStructureEvidence"]
    assert "Visited Array" not in evidence_map["dataStructureEvidence"]
    assert "Visited Array" in evidence_map["techniqueEvidence"]
    assert "State Tracking" in evidence_map["techniqueEvidence"]
    assert evidence_map["commonMistakes"] == [
        "忘記標記 visited。",
        "queue 初始化錯誤，導致起點或距離沒有被正確設定。",
    ]


def test_online_pipeline_suppresses_partial_title_seed_for_python_input():
    result = OnlineQueryPipeline(documents=(_uva_document(),)).run(
        "def solve():\n    bombs = mines = shortest = path = []",
        top_k=1,
    )
    trace = result.trace.to_mapping()

    assert result.query_understanding.input_kind == "python"
    assert result.matched_problem is None
    assert trace["matchedProblem"] is None


def test_online_pipeline_suppresses_partial_title_seed_for_cpp_input():
    result = OnlineQueryPipeline(documents=(_uva_document(),)).run(
        """
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<int> bombs, mines, shortest, path;
    return 0;
}
""".strip(),
        top_k=1,
    )
    trace = result.trace.to_mapping()

    assert result.query_understanding.input_kind == "cpp"
    assert result.matched_problem is None
    assert trace["matchedProblem"] is None

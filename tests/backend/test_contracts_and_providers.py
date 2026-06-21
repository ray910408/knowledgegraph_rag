from __future__ import annotations

import math

from backend.app.contracts import (
    EntityRecord,
    ProblemChunk,
    RawProblem,
    RelationRecord,
    RetrievalEvidenceBundle,
    RetrievalTrace,
)
from backend.app.providers import DeterministicMockEmbeddingProvider
from backend.app.stores import (
    BM25Document,
    BM25Store,
    GraphStore,
    SearchCandidate,
    VectorRecord,
    VectorStore,
)


def test_raw_problem_contract_round_trips_existing_and_extended_fields():
    raw = {
        "id": "leetcode-994",
        "source": "LeetCode",
        "sourceId": "994",
        "title": "Rotting Oranges",
        "problemType": "Graph Traversal",
        "statement": "Find the minimum minutes until no fresh orange remains.",
        "answer": "Use multi-source BFS.",
        "solutionHints": ["Put all rotten oranges in the queue first."],
        "concepts": ["BFS", "Queue"],
        "tags": ["matrix", "graph"],
        "metadata": {"url": "https://leetcode.com/problems/rotting-oranges/"},
        "difficulty": "Medium",
        "constraints": ["1 <= m, n <= 10"],
        "examples": [{"input": "grid", "output": "4"}],
        "editorial": "Multi-source BFS from initially rotten oranges.",
    }

    problem = RawProblem.from_mapping(raw)
    assert problem.source_id == "994"
    assert problem.problem_type == "Graph Traversal"
    assert problem.solution_hints == ("Put all rotten oranges in the queue first.",)
    assert problem.difficulty == "Medium"
    assert problem.examples == ({"input": "grid", "output": "4"},)

    assert problem.to_mapping() == raw


def test_ingestion_artifact_records_are_serializable():
    chunk = ProblemChunk(
        id="leetcode-994:statement:0",
        problem_id="leetcode-994",
        kind="statement",
        text="Find the minimum minutes.",
        index=0,
        concepts=("BFS",),
        metadata={"source": "LeetCode"},
    )
    entity = EntityRecord(
        id="concept:bfs",
        name="BFS",
        type="algorithm",
        aliases=("Breadth First Search",),
        problem_ids=("leetcode-994",),
        metadata={"origin": "mock-extractor"},
    )
    relation = RelationRecord(
        id="leetcode-994->concept:bfs",
        source_id="leetcode-994",
        target_id="concept:bfs",
        type="REQUIRES",
        weight=1.0,
        evidence=("problem statement mentions breadth-first expansion",),
        metadata={"origin": "mock-extractor"},
    )

    assert chunk.to_mapping()["problemId"] == "leetcode-994"
    assert entity.to_mapping()["aliases"] == ["Breadth First Search"]
    assert relation.to_mapping()["sourceId"] == "leetcode-994"


def test_retrieval_trace_and_evidence_bundle_have_expected_debug_shape():
    trace = RetrievalTrace(
        query_understanding={"intent": "problem_search"},
        entity_linking=[{"entityId": "concept:bfs", "name": "BFS"}],
        vector_candidates=[{"id": "leetcode-994", "score": 0.9}],
        graph_candidates=[{"id": "leetcode-1091", "score": 0.8}],
        bm25_candidates=[{"id": "uva-10653", "score": 2.0}],
        fusion_scores=[{"id": "leetcode-994", "score": 1.0}],
        reranker_scores=[{"id": "leetcode-994", "score": 0.95}],
    )
    bundle = RetrievalEvidenceBundle(
        similar_problems=[{"id": "leetcode-994"}],
        graph_paths=[{"nodes": ["input", "concept:bfs", "leetcode-994"]}],
        algorithm_evidence=["BFS"],
        data_structure_evidence=["Queue"],
        pattern_evidence=["Graph Traversal"],
        common_mistakes=["forget visited"],
    )

    assert set(trace.to_mapping()) == {
        "queryUnderstanding",
        "entityLinking",
        "vectorCandidates",
        "graphCandidates",
        "bm25Candidates",
        "fusionScores",
        "rerankerScores",
    }
    assert bundle.to_mapping()["commonMistakes"] == ["forget visited"]


def test_deterministic_mock_embedding_provider_is_stable_and_normalized():
    provider = DeterministicMockEmbeddingProvider(dimension=8)

    first = provider.embed_text("BFS shortest path")
    second = provider.embed_text("BFS shortest path")
    different = provider.embed_text("dynamic programming")

    assert provider.model_name == "BAAI/bge-m3"
    assert first == second
    assert first != different
    assert len(first) == 8
    assert math.isclose(sum(value * value for value in first), 1.0, rel_tol=1e-6)
    assert provider.embed_batch(["BFS shortest path", "BFS shortest path"]) == [first, first]


def test_store_protocols_are_structural_contracts():
    class MemoryVectorStore:
        def upsert(self, records):
            self.records = tuple(records)

        def search(self, query_vector, *, top_k, filters=None):
            return (SearchCandidate(id="leetcode-994", score=0.9, payload={"source": "mock"}),)

    class MemoryBM25Store:
        def index_documents(self, documents):
            self.documents = tuple(documents)

        def search(self, query, *, top_k):
            return (SearchCandidate(id="leetcode-994", score=2.0, payload={}),)

    class MemoryGraphStore:
        def upsert_entities(self, entities):
            self.entities = tuple(entities)

        def upsert_relations(self, relations):
            self.relations = tuple(relations)

        def find_paths(self, source_id, target_id, *, max_hops=3):
            return ({"nodes": [source_id, "concept:bfs", target_id]},)

    vector_store = MemoryVectorStore()
    bm25_store = MemoryBM25Store()
    graph_store = MemoryGraphStore()

    assert isinstance(vector_store, VectorStore)
    assert isinstance(bm25_store, BM25Store)
    assert isinstance(graph_store, GraphStore)

    vector_store.upsert((VectorRecord(id="chunk-1", vector=(0.1, 0.2), payload={}),))
    bm25_store.index_documents((BM25Document(id="chunk-1", text="BFS shortest path"),))
    graph_store.upsert_entities((EntityRecord(id="concept:bfs", name="BFS", type="algorithm"),))

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Sequence

import pytest

from backend.app.providers import DeterministicMockEmbeddingProvider
from backend.app.retrieval.pipeline import RetrievalDocument
from backend.app.stores import BM25Document, SearchCandidate


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


def _write_bm25_index(path: Path) -> None:
    path.write_text(
        """
{
  "documents": [
    {
      "id": "leetcode-994:statement:0",
      "text": "Multi-source BFS with a queue on a grid.",
      "problemId": "leetcode-994",
      "payload": {
        "problemId": "leetcode-994",
        "kind": "statement",
        "text": "Multi-source BFS with a queue on a grid.",
        "concepts": ["BFS", "Queue"],
        "metadata": {
          "source": "LeetCode",
          "sourceId": "994",
          "title": "Rotting Oranges",
          "problemType": "Graph Traversal"
        }
      }
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )


class FakeVectorStore:
    constructor_calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, *, client=None, collection_name: str, url: str, timeout: float = 1.0):
        self.collection_name = collection_name
        self.url = url
        self.constructor_calls.append(
            {
                "client": client,
                "collection_name": collection_name,
                "timeout": timeout,
                "url": url,
            }
        )

    def upsert(self, records: Sequence[object]) -> None:
        raise AssertionError("runtime search should not upsert vector records")

    def search(self, query_vector: Sequence[float], *, top_k: int, filters=None):
        return (
            SearchCandidate(
                id="leetcode-994:statement:0",
                score=0.99,
                payload={
                    "problemId": "leetcode-994",
                    "kind": "statement",
                    "text": "Multi-source BFS with a queue on a grid.",
                    "concepts": ["BFS", "Queue"],
                    "metadata": {
                        "source": "LeetCode",
                        "sourceId": "994",
                        "title": "Rotting Oranges",
                        "problemType": "Graph Traversal",
                    },
                },
            ),
        )


class FakeGraphStore:
    constructor_calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, *, driver=None, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.constructor_calls.append(
            {
                "driver": driver,
                "password": password,
                "uri": uri,
                "user": user,
            }
        )

    def upsert_entities(self, entities: Sequence[object]) -> None:
        raise AssertionError("runtime search should not upsert graph entities")

    def upsert_relations(self, relations: Sequence[object]) -> None:
        raise AssertionError("runtime search should not upsert graph relations")

    def find_paths(self, source_id: str, target_id: str, *, max_hops: int = 3):
        if source_id == "leetcode-994" and target_id in {"concept:bfs", "concept:queue"}:
            return (
                {
                    "nodes": [source_id, target_id],
                    "relations": ["REQUIRES"],
                    "score": 1.0,
                },
            )
        return ()


def test_load_runtime_retrieval_settings_defaults_to_local():
    from backend.app.retrieval.runtime import load_runtime_retrieval_settings

    settings = load_runtime_retrieval_settings({})

    assert settings.backend == "local"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "programming_chunks"
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password == "password"
    assert settings.bm25_index_path.name == "bm25_index.json"


def test_load_runtime_retrieval_settings_rejects_unknown_backend():
    from backend.app.retrieval.runtime import RuntimeRetrievalError, load_runtime_retrieval_settings

    with pytest.raises(RuntimeRetrievalError, match="unsupported RETRIEVAL_BACKEND"):
        load_runtime_retrieval_settings({"RETRIEVAL_BACKEND": "remote"})


def test_json_bm25_store_loads_processed_index(tmp_path):
    from backend.app.retrieval.runtime import JsonBM25Store

    index_path = tmp_path / "bm25_index.json"
    _write_bm25_index(index_path)

    store = JsonBM25Store.from_path(index_path)
    results = store.search("BFS queue shortest path", top_k=1)

    assert results[0].id == "leetcode-994:statement:0"
    assert results[0].payload["problemId"] == "leetcode-994"
    assert results[0].payload["metadata"]["title"] == "Rotting Oranges"


def test_json_bm25_store_can_accept_runtime_documents_after_loading(tmp_path):
    from backend.app.retrieval.runtime import JsonBM25Store

    index_path = tmp_path / "bm25_index.json"
    _write_bm25_index(index_path)
    store = JsonBM25Store.from_path(index_path)
    store.index_documents((BM25Document(id="extra", text="binary search", payload={"problemId": "extra"}),))

    results = store.search("binary search", top_k=1)

    assert results[0].id == "extra"
    assert results[0].payload["problemId"] == "extra"


def test_build_runtime_retrieval_local_does_not_construct_external_stores(monkeypatch):
    from backend.app.retrieval import runtime

    def fail_qdrant(**kwargs):
        raise AssertionError("local mode must not construct Qdrant")

    def fail_neo4j(**kwargs):
        raise AssertionError("local mode must not construct Neo4j")

    monkeypatch.setattr(runtime, "QdrantVectorStore", fail_qdrant)
    monkeypatch.setattr(runtime, "Neo4jGraphStore", fail_neo4j)
    settings = runtime.RuntimeRetrievalSettings(backend="local")

    configured = runtime.build_runtime_retrieval(
        settings=settings,
        documents=_documents(),
        embedding_provider=DeterministicMockEmbeddingProvider(dimension=8),
    )
    result = configured.pipeline.run("BFS queue shortest path", top_k=2)

    assert configured.backend == "local"
    assert configured.candidate_sources == {"vector": "local", "graph": "local", "bm25": "local"}
    assert result.vector_candidates
    assert result.bm25_candidates


def test_build_runtime_retrieval_stores_injects_qdrant_neo4j_and_bm25(monkeypatch, tmp_path):
    from backend.app.retrieval import runtime

    index_path = tmp_path / "bm25_index.json"
    _write_bm25_index(index_path)
    FakeVectorStore.constructor_calls.clear()
    FakeGraphStore.constructor_calls.clear()
    monkeypatch.setattr(runtime, "QdrantVectorStore", FakeVectorStore)
    monkeypatch.setattr(runtime, "Neo4jGraphStore", FakeGraphStore)
    settings = runtime.RuntimeRetrievalSettings(
        backend="stores",
        qdrant_url="http://qdrant.example:6333",
        qdrant_collection="programming_chunks",
        neo4j_uri="bolt://neo4j.example:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        bm25_index_path=index_path,
    )

    configured = runtime.build_runtime_retrieval(
        settings=settings,
        documents=_documents(),
        embedding_provider=DeterministicMockEmbeddingProvider(dimension=8),
    )
    result = configured.pipeline.run("BFS queue graph traversal", top_k=2)
    trace = result.trace.to_mapping()

    assert configured.backend == "stores"
    assert configured.candidate_sources == {
        "vector": "qdrant",
        "graph": "neo4j",
        "bm25": "bm25_index",
    }
    assert FakeVectorStore.constructor_calls
    assert FakeGraphStore.constructor_calls
    vector_call = FakeVectorStore.constructor_calls[0]
    graph_call = FakeGraphStore.constructor_calls[0]
    assert vector_call["url"] == "http://qdrant.example:6333"
    assert vector_call["collection_name"] == "programming_chunks"
    assert graph_call["uri"] == "bolt://neo4j.example:7687"
    assert graph_call["user"] == "neo4j"
    assert graph_call["password"] == "password"
    assert trace["vectorCandidates"][0]["payload"]["storeCandidateId"] == "leetcode-994:statement:0"
    assert trace["bm25Candidates"][0]["payload"]["storeCandidateId"] == "leetcode-994:statement:0"
    assert trace["vectorCandidates"][0]["payload"]["answer"] == ""
    assert trace["bm25Candidates"][0]["payload"]["answer"] == ""
    assert trace["graphCandidates"][0]["id"] == "leetcode-994"
    assert any("storePath" in path for path in result.graph_paths)


def test_add_runtime_debug_trace_labels_candidate_sources():
    from backend.app.retrieval.runtime import add_runtime_debug_trace

    trace = {
        "vectorCandidates": [{"id": "v"}],
        "graphCandidates": [{"id": "g"}],
        "bm25Candidates": [{"id": "b"}],
        "fusionScores": [{"id": "h"}],
        "rerankerScores": [{"id": "r"}],
    }

    labeled = add_runtime_debug_trace(
        trace,
        {"vector": "qdrant", "graph": "neo4j", "bm25": "bm25_index"},
    )

    assert labeled["candidateSources"] == {
        "vector": "qdrant",
        "graph": "neo4j",
        "bm25": "bm25_index",
    }
    assert labeled["vectorCandidates"][0]["candidateSource"] == "qdrant"
    assert labeled["graphCandidates"][0]["candidateSource"] == "neo4j"
    assert labeled["bm25Candidates"][0]["candidateSource"] == "bm25_index"
    assert labeled["fusionScores"][0]["candidateSource"] == "hybrid"
    assert labeled["rerankerScores"][0]["candidateSource"] == "hybrid"

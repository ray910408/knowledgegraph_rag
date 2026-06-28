from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar, Sequence

import pytest

from backend.app.contracts import ProblemChunk
from backend.app.ingestion.pipeline import (
    _classify_concept,
    _write_bm25_index as write_ingestion_bm25_index,
)
from backend.app.providers import DeterministicMockEmbeddingProvider
from backend.app.retrieval.pipeline import QueryUnderstandingService
from backend.app.retrieval.pipeline import RetrievalDocument
from backend.app.stores import BM25Document, SearchCandidate

CHINESE_SHORTEST_PATH_QUERY = "給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。"


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


def _write_processed_problems(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "leetcode-994",
                    "source": "LeetCode",
                    "sourceId": "994",
                    "title": "Processed Rotting Oranges",
                    "problemType": "Graph Traversal",
                    "statement": "Processed statement for BFS queue traversal.",
                    "answer": "Processed answer: run multi-source BFS.",
                    "solutionHints": [
                        "Processed hint: enqueue all rotten oranges first.",
                    ],
                    "concepts": ["BFS", "Queue"],
                    "metadata": {"difficulty": "medium"},
                    "constraints": ["1 <= grid.length <= 10"],
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
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

    def compatibility_warning(self):
        return None


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
    assert settings.processed_problems_path.name == "problems.json"


def test_load_runtime_retrieval_settings_reads_processed_problem_path(tmp_path):
    from backend.app.retrieval.runtime import load_runtime_retrieval_settings

    processed_path = tmp_path / "custom-problems.json"

    settings = load_runtime_retrieval_settings(
        {
            "RETRIEVAL_BACKEND": "stores",
            "PROCESSED_PROBLEMS_PATH": str(processed_path),
        }
    )

    assert settings.backend == "stores"
    assert settings.processed_problems_path == processed_path

    relative_settings = load_runtime_retrieval_settings(
        {
            "RETRIEVAL_BACKEND": "stores",
            "PROCESSED_PROBLEMS_PATH": "custom/problems.json",
        }
    )

    repo_root = Path(__file__).resolve().parents[2]
    assert relative_settings.backend == "stores"
    assert relative_settings.processed_problems_path == repo_root / "custom" / "problems.json"


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


def test_json_bm25_store_matches_problem_alias_text(tmp_path):
    from backend.app.retrieval.runtime import JsonBM25Store

    index_path = tmp_path / "bm25_index.json"
    chunk = ProblemChunk(
        id="uva-10653:statement:0",
        problem_id="uva-10653",
        kind="statement",
        text="Find the shortest safe path on a grid with bomb cells.",
        index=0,
        concepts=("BFS", "Queue", "Visited Array"),
        metadata={
            "source": "UVa",
            "sourceId": "10653",
            "title": "Bombs! NO they are Mines!!",
            "problemType": "Graph Traversal",
        },
        source="UVa",
        source_id="10653",
        title="Bombs! NO they are Mines!!",
        problem_type="Graph Traversal",
    )
    write_ingestion_bm25_index(index_path, (chunk,))

    artifact = json.loads(index_path.read_text(encoding="utf-8"))
    search_text = artifact["documents"][0]["text"]
    assert search_text.startswith(
        "uva-10653 UVa 10653 UVa-10653 UVa 10653 "
        "Bombs! NO they are Mines!! Graph Traversal "
        "BFS Queue Visited Array "
    )
    for needle in (
        "graph traversal",
        "圖論遍歷",
        "圖遍歷",
        "廣搜",
        "廣度優先搜尋",
        "佇列",
        "拜訪陣列",
    ):
        assert needle in search_text
    assert search_text.endswith("Find the shortest safe path on a grid with bomb cells.")

    store = JsonBM25Store.from_path(index_path)
    results = store.search("UVA-10653 - Bombs! NO they are Mines!!", top_k=1)

    assert results[0].id == chunk.id
    assert results[0].score > 0


def test_json_bm25_store_matches_multilingual_bm25_query_variant(tmp_path):
    from backend.app.retrieval.runtime import JsonBM25Store

    index_path = tmp_path / "bm25_index.json"
    _write_bm25_index(index_path)

    understanding = QueryUnderstandingService().understand(CHINESE_SHORTEST_PATH_QUERY)
    store = JsonBM25Store.from_path(index_path)
    results = store.search(understanding.query_variants["bm25"], top_k=1)

    assert results[0].id == "leetcode-994:statement:0"
    assert results[0].score > 0


@pytest.mark.parametrize(
    ("concept", "expected"),
    (
        ("BFS", "algorithm"),
        ("DFS", "algorithm"),
        ("Dijkstra", "algorithm"),
        ("Binary Search", "algorithm"),
        ("Dynamic Programming", "algorithm"),
        ("Queue", "data_structure"),
        ("Stack", "data_structure"),
        ("Heap", "data_structure"),
        ("Array", "data_structure"),
        ("Hash Map", "data_structure"),
        ("Visited Array", "technique"),
        ("Visited Set", "technique"),
        ("State Tracking", "technique"),
        ("Shortest Path", "concept"),
    ),
)
def test_ingestion_classifies_concepts_for_graph_metadata(concept, expected):
    assert _classify_concept(concept) == expected


def test_json_bm25_store_rejects_non_object_index(tmp_path):
    from backend.app.retrieval.runtime import JsonBM25Store, RuntimeRetrievalError

    index_path = tmp_path / "bm25_index.json"
    index_path.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeRetrievalError, match="documents list"):
        JsonBM25Store.from_path(index_path)


@pytest.mark.parametrize(
    "document",
    (
        {"text": "missing id"},
        {"id": "bad-payload", "payload": []},
    ),
)
def test_json_bm25_store_rejects_malformed_document_entries(tmp_path, document):
    from backend.app.retrieval.runtime import JsonBM25Store, RuntimeRetrievalError

    index_path = tmp_path / "bm25_index.json"
    index_path.write_text(
        json.dumps({"documents": [document]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeRetrievalError, match="BM25 document"):
        JsonBM25Store.from_path(index_path)


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

    embedding_provider = DeterministicMockEmbeddingProvider(dimension=8)
    configured = runtime.build_runtime_retrieval(
        settings=settings,
        documents=_documents(),
        embedding_provider=embedding_provider,
    )
    result = configured.pipeline.run("BFS queue shortest path", top_k=2)

    assert configured.backend == "local"
    assert configured.candidate_sources == {"vector": "local", "graph": "local", "bm25": "local"}
    assert configured.pipeline._embedding_provider is embedding_provider
    assert configured.provider_sources == {
        "embedding": {
            "provider": "mock",
            "model": "BAAI/bge-m3",
        },
        "reranker": {
            "provider": "mock",
            "model": "BAAI/bge-reranker-v2-m3",
        },
    }
    assert configured.compatibility_warnings == []
    assert result.vector_candidates
    assert result.bm25_candidates


def test_build_runtime_retrieval_local_ignores_missing_processed_problems(monkeypatch, tmp_path):
    from backend.app.retrieval import runtime

    class FailingProcessedProblemDocumentLoader:
        def __init__(self, path: Path) -> None:
            raise AssertionError(f"local mode must not construct processed loader for {path}")

    missing_processed_path = tmp_path / "missing-problems.json"
    monkeypatch.setattr(
        runtime,
        "ProcessedProblemDocumentLoader",
        FailingProcessedProblemDocumentLoader,
    )
    settings = runtime.load_runtime_retrieval_settings(
        {"PROCESSED_PROBLEMS_PATH": str(missing_processed_path)}
    )

    embedding_provider = DeterministicMockEmbeddingProvider(dimension=8)
    configured = runtime.build_runtime_retrieval(
        settings=settings,
        embedding_provider=embedding_provider,
    )
    result = configured.pipeline.run("BFS queue shortest path", top_k=2)

    assert settings.backend == "local"
    assert settings.processed_problems_path == missing_processed_path
    assert not missing_processed_path.exists()
    assert configured.backend == "local"
    assert configured.candidate_sources == {"vector": "local", "graph": "local", "bm25": "local"}
    assert configured.pipeline._embedding_provider is embedding_provider
    assert result.vector_candidates
    assert result.bm25_candidates


def test_build_runtime_retrieval_rejects_invalid_settings_backend(monkeypatch):
    from backend.app.retrieval import runtime

    def fail_qdrant(**kwargs):
        raise AssertionError("invalid backend must not construct Qdrant")

    def fail_neo4j(**kwargs):
        raise AssertionError("invalid backend must not construct Neo4j")

    monkeypatch.setattr(runtime, "QdrantVectorStore", fail_qdrant)
    monkeypatch.setattr(runtime, "Neo4jGraphStore", fail_neo4j)
    settings = runtime.RuntimeRetrievalSettings(backend="bad")  # type: ignore[arg-type]

    with pytest.raises(runtime.RuntimeRetrievalError, match="unsupported retrieval backend"):
        runtime.build_runtime_retrieval(
            settings=settings,
            documents=_documents(),
            embedding_provider=DeterministicMockEmbeddingProvider(dimension=8),
        )


def test_build_runtime_retrieval_stores_injects_qdrant_neo4j_and_bm25(monkeypatch, tmp_path):
    from backend.app.retrieval import runtime

    class FailingProcessedProblemDocumentLoader:
        def __init__(self, path: Path) -> None:
            raise AssertionError(f"explicit documents must not construct loader for {path}")

    index_path = tmp_path / "bm25_index.json"
    _write_bm25_index(index_path)
    FakeVectorStore.constructor_calls.clear()
    FakeGraphStore.constructor_calls.clear()
    monkeypatch.setattr(runtime, "QdrantVectorStore", FakeVectorStore)
    monkeypatch.setattr(runtime, "Neo4jGraphStore", FakeGraphStore)
    monkeypatch.setattr(runtime, "ProcessedProblemDocumentLoader", FailingProcessedProblemDocumentLoader)
    settings = runtime.RuntimeRetrievalSettings(
        backend="stores",
        qdrant_url="http://qdrant.example:6333",
        qdrant_collection="custom_runtime_chunks",
        neo4j_uri="bolt://neo4j.example:7687",
        neo4j_user="custom_user",
        neo4j_password="custom_password",
        bm25_index_path=index_path,
        processed_problems_path=tmp_path / "missing-problems.json",
    )

    embedding_provider = DeterministicMockEmbeddingProvider(dimension=8)
    configured = runtime.build_runtime_retrieval(
        settings=settings,
        documents=_documents(),
        embedding_provider=embedding_provider,
    )
    result = configured.pipeline.run("BFS queue graph traversal", top_k=2)
    trace = result.trace.to_mapping()

    assert configured.backend == "stores"
    assert configured.candidate_sources == {
        "vector": "qdrant",
        "graph": "neo4j",
        "bm25": "bm25_index",
    }
    assert configured.pipeline._embedding_provider is embedding_provider
    assert configured.provider_sources == {
        "embedding": {
            "provider": "mock",
            "model": "BAAI/bge-m3",
            "adapter": "qdrant",
        },
        "reranker": {
            "provider": "mock",
            "model": "BAAI/bge-reranker-v2-m3",
        },
    }
    assert len(FakeVectorStore.constructor_calls) == 1
    assert len(FakeGraphStore.constructor_calls) == 1
    vector_call = FakeVectorStore.constructor_calls[0]
    graph_call = FakeGraphStore.constructor_calls[0]
    assert vector_call["url"] == "http://qdrant.example:6333"
    assert vector_call["collection_name"] == "custom_runtime_chunks"
    assert graph_call["uri"] == "bolt://neo4j.example:7687"
    assert graph_call["user"] == "custom_user"
    assert graph_call["password"] == "custom_password"
    assert trace["vectorCandidates"][0]["payload"]["storeCandidateId"] == "leetcode-994:statement:0"
    assert trace["bm25Candidates"][0]["payload"]["storeCandidateId"] == "leetcode-994:statement:0"
    assert trace["vectorCandidates"][0]["payload"]["answer"] == ""
    assert trace["bm25Candidates"][0]["payload"]["answer"] == ""
    assert trace["graphCandidates"][0]["id"] == "leetcode-994"
    assert any("storePath" in path for path in result.graph_paths)


def test_build_runtime_retrieval_stores_propagates_qdrant_compatibility_warning(monkeypatch, tmp_path):
    from backend.app.retrieval import runtime

    class WarningVectorStore(FakeVectorStore):
        def compatibility_warning(self):
            return {
                "adapter": "qdrant",
                "severity": "warning",
                "message": "Qdrant client 1.18.0 is outside the supported server 1.15.3 minor range.",
            }

    index_path = tmp_path / "bm25_index.json"
    _write_bm25_index(index_path)
    monkeypatch.setattr(runtime, "QdrantVectorStore", WarningVectorStore)
    monkeypatch.setattr(runtime, "Neo4jGraphStore", FakeGraphStore)
    settings = runtime.RuntimeRetrievalSettings(backend="stores", bm25_index_path=index_path)

    configured = runtime.build_runtime_retrieval(
        settings=settings,
        documents=_documents(),
        embedding_provider=DeterministicMockEmbeddingProvider(dimension=8),
    )

    assert configured.compatibility_warnings == [
        {
            "adapter": "qdrant",
            "severity": "warning",
            "message": "Qdrant client 1.18.0 is outside the supported server 1.15.3 minor range.",
        }
    ]


def test_build_runtime_retrieval_stores_ignores_compatibility_hook_errors(monkeypatch, tmp_path):
    from backend.app.retrieval import runtime

    class ExplodingWarningVectorStore(FakeVectorStore):
        def compatibility_warning(self):
            raise RuntimeError("compatibility probe unavailable")

    index_path = tmp_path / "bm25_index.json"
    _write_bm25_index(index_path)
    monkeypatch.setattr(runtime, "QdrantVectorStore", ExplodingWarningVectorStore)
    monkeypatch.setattr(runtime, "Neo4jGraphStore", FakeGraphStore)
    settings = runtime.RuntimeRetrievalSettings(backend="stores", bm25_index_path=index_path)

    configured = runtime.build_runtime_retrieval(
        settings=settings,
        documents=_documents(),
        embedding_provider=DeterministicMockEmbeddingProvider(dimension=8),
    )

    assert configured.backend == "stores"
    assert configured.compatibility_warnings == []


def test_build_runtime_retrieval_stores_loads_processed_documents_when_no_override(monkeypatch, tmp_path):
    from backend.app.retrieval import runtime

    index_path = tmp_path / "bm25_index.json"
    processed_path = tmp_path / "problems.json"
    _write_bm25_index(index_path)
    _write_processed_problems(processed_path)
    FakeVectorStore.constructor_calls.clear()
    FakeGraphStore.constructor_calls.clear()
    monkeypatch.setattr(runtime, "QdrantVectorStore", FakeVectorStore)
    monkeypatch.setattr(runtime, "Neo4jGraphStore", FakeGraphStore)
    settings = runtime.RuntimeRetrievalSettings(
        backend="stores",
        bm25_index_path=index_path,
        processed_problems_path=processed_path,
    )

    configured = runtime.build_runtime_retrieval(
        settings=settings,
        embedding_provider=DeterministicMockEmbeddingProvider(dimension=8),
    )
    result = configured.pipeline.run("BFS queue graph traversal", top_k=2)

    assert configured.backend == "stores"
    assert result.graph_candidates[0].title == "Processed Rotting Oranges"
    assert result.graph_candidates[0].payload["answer"] == "Processed answer: run multi-source BFS."


def test_add_runtime_debug_trace_labels_candidate_sources():
    from backend.app.retrieval.runtime import add_runtime_debug_trace

    trace = {
        "vectorCandidates": [{"id": "v", "source": "vector"}],
        "graphCandidates": [{"id": "g", "source": "graph"}],
        "bm25Candidates": [{"id": "b", "source": "bm25"}],
        "fusionScores": [{"id": "h", "source": "hybrid"}],
        "rerankerScores": [{"id": "r", "source": "hybrid"}],
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
    assert labeled["vectorCandidates"][0]["source"] == "vector"
    assert labeled["graphCandidates"][0]["source"] == "graph"
    assert labeled["bm25Candidates"][0]["source"] == "bm25"
    assert labeled["fusionScores"][0]["source"] == "hybrid"
    assert labeled["rerankerScores"][0]["source"] == "hybrid"


def test_add_runtime_debug_trace_copies_provider_sources_without_mutating_inputs():
    from backend.app.retrieval.runtime import add_runtime_debug_trace

    trace = {
        "vectorCandidates": [{"id": "v"}],
        "graphCandidates": [],
        "bm25Candidates": [],
        "fusionScores": [],
        "rerankerScores": [],
    }
    candidate_sources = {"vector": "local", "graph": "local", "bm25": "local"}
    provider_sources = {
        "embedding": {"provider": "mock", "model": "BAAI/bge-m3"},
        "reranker": {
            "provider": "mock",
            "model": "BAAI/bge-reranker-v2-m3",
        },
    }

    labeled = add_runtime_debug_trace(
        trace,
        candidate_sources,
        provider_sources,
    )
    labeled["providerSources"]["embedding"]["model"] = "changed"

    assert trace == {
        "vectorCandidates": [{"id": "v"}],
        "graphCandidates": [],
        "bm25Candidates": [],
        "fusionScores": [],
        "rerankerScores": [],
    }
    assert candidate_sources == {"vector": "local", "graph": "local", "bm25": "local"}
    assert provider_sources["embedding"]["model"] == "BAAI/bge-m3"


def test_runtime_debug_trace_includes_store_compatibility_warnings():
    from backend.app.retrieval.runtime import add_runtime_debug_trace

    warning = {
        "adapter": "qdrant",
        "severity": "warning",
        "message": "Qdrant client 1.18.0 is outside the supported server 1.15.3 minor range.",
    }
    labeled = add_runtime_debug_trace(
        {
            "vectorCandidates": [],
            "graphCandidates": [],
            "bm25Candidates": [],
            "fusionScores": [],
            "rerankerScores": [],
        },
        {"vector": "qdrant", "graph": "neo4j", "bm25": "bm25_index"},
        compatibility_warnings=(warning,),
    )

    assert labeled["compatibilityWarnings"] == [
        {
            "adapter": "qdrant",
            "severity": "warning",
            "message": "Qdrant client 1.18.0 is outside the supported server 1.15.3 minor range.",
        }
    ]
    labeled["compatibilityWarnings"][0]["message"] = "changed"
    assert warning == {
        "adapter": "qdrant",
        "severity": "warning",
        "message": "Qdrant client 1.18.0 is outside the supported server 1.15.3 minor range.",
    }


@pytest.mark.parametrize(
    ("client_version", "server_version", "expected"),
    [
        ("1.15.4", "1.15.3", None),
        (
            "1.18.0",
            "1.15.3",
            {
                "adapter": "qdrant",
                "severity": "warning",
                "message": "Qdrant client 1.18.0 is outside the supported server 1.15.3 minor range.",
            },
        ),
        (None, "1.15.3", None),
        ("1.18.0", None, None),
    ],
)
def test_qdrant_compatibility_warning_handles_versions(client_version, server_version, expected):
    from backend.app.adapters.qdrant import qdrant_compatibility_warning

    class FakeClient:
        def get_version(self):
            return server_version

    client = FakeClient()
    client.client_version = client_version

    assert qdrant_compatibility_warning(client) == expected


def test_qdrant_compatibility_warning_ignores_get_version_errors():
    from backend.app.adapters.qdrant import qdrant_compatibility_warning

    class FakeClient:
        client_version = "1.18.0"

        def get_version(self):
            raise RuntimeError("server unavailable")

    assert qdrant_compatibility_warning(FakeClient()) is None


def test_qdrant_compatibility_warning_uses_version_attribute_fallback():
    from backend.app.adapters.qdrant import qdrant_compatibility_warning

    class FakeClient:
        version = "1.18.0"

        def get_version(self):
            return "1.15.3"

    assert qdrant_compatibility_warning(FakeClient()) == {
        "adapter": "qdrant",
        "severity": "warning",
        "message": "Qdrant client 1.18.0 is outside the supported server 1.15.3 minor range.",
    }


def test_qdrant_compatibility_warning_uses_package_version_and_public_info(monkeypatch):
    from backend.app.adapters import qdrant

    package_names: list[str] = []

    def fake_package_version(package_name: str) -> str:
        package_names.append(package_name)
        return "1.18.0"

    monkeypatch.setattr(qdrant.metadata, "version", fake_package_version)

    class VersionInfo:
        version = "1.15.3"

    class Sdk118Client:
        def __init__(self):
            self.info_calls = 0

        def info(self):
            self.info_calls += 1
            return VersionInfo()

    client = Sdk118Client()

    assert qdrant.qdrant_compatibility_warning(client) == {
        "adapter": "qdrant",
        "severity": "warning",
        "message": "Qdrant client 1.18.0 is outside the supported server 1.15.3 minor range.",
    }
    assert package_names == ["qdrant-client"]
    assert client.info_calls == 1


def test_qdrant_vector_store_disables_sdk_compatibility_check(monkeypatch):
    import qdrant_client

    from backend.app.adapters.qdrant import QdrantVectorStore

    captured: dict[str, object] = {}

    class FakeQdrantClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrantClient)

    QdrantVectorStore(url="http://qdrant.example:6333", timeout=2.0)

    assert captured["url"] == "http://qdrant.example:6333"
    assert captured["timeout"] == 2.0
    assert captured["check_compatibility"] is False


def test_qdrant_compatibility_warning_ignores_client_version_property_errors():
    from backend.app.adapters.qdrant import qdrant_compatibility_warning

    class FakeClient:
        @property
        def client_version(self):
            raise RuntimeError("client version unavailable")

        version = "1.18.0"

        def get_version(self):
            return "1.15.3"

    assert qdrant_compatibility_warning(FakeClient()) is None


def test_add_runtime_debug_trace_omits_provider_sources_when_not_supplied():
    from backend.app.retrieval.runtime import add_runtime_debug_trace

    labeled = add_runtime_debug_trace(
        {
            "vectorCandidates": [],
            "graphCandidates": [],
            "bm25Candidates": [],
            "fusionScores": [],
            "rerankerScores": [],
        },
        {"vector": "local", "graph": "local", "bm25": "local"},
    )

    assert "providerSources" not in labeled

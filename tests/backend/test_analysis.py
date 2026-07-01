from collections.abc import Iterator
import json

import pytest
from fastapi.testclient import TestClient

from backend.app.analysis import (
    detect_graph_traversal_signals,
    find_graph_traversal_examples,
    load_programming_dataset,
)
from backend.app.contracts import RetrievalTrace
from backend.app.main import (
    AnalysisRequest,
    _analysis_paths_from_graph_trace,
    _filter_retrieval_trace,
    _retrieval_problem_type,
    app,
)
from backend.app.providers import DeterministicMockEmbeddingProvider
from backend.app.retrieval.pipeline import (
    ExactProblemMatch,
    OnlineQueryPipeline,
    OnlineQueryResult,
    EntityLinkingService,
    QueryUnderstanding,
    QueryUnderstandingService,
    RetrievalCandidate,
    RetrievalDocument,
)
from backend.app.retrieval.runtime import RuntimeRetrieval


@pytest.fixture(autouse=True)
def isolate_runtime_retrieval() -> Iterator[None]:
    _clear_runtime_retrieval()
    yield
    _clear_runtime_retrieval()


def _clear_runtime_retrieval() -> None:
    try:
        delattr(app.state, "runtime_retrieval")
    except (AttributeError, KeyError):
        pass


def _required_concept_names(payload: dict[str, object]) -> set[str]:
    return {
        str(concept["name"])
        for concept in payload["requiredConcepts"]  # type: ignore[index]
    }


def _similar_problem_ids(payload: dict[str, object]) -> set[str]:
    return {
        str(problem["id"])
        for problem in payload["similarProblems"]  # type: ignore[index]
    }


def _id_source_pairs(records: list[dict[str, object]]) -> list[tuple[str, str]]:
    return [
        (str(record["id"]), str(record.get("sourceId") or ""))
        for record in records
        if record.get("id") is not None
    ]


def _response_text(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def test_analysis_preserves_candidate_source_and_snake_case_source_id():
    candidate = RetrievalCandidate(
        id="qdrant-123",
        title="Qdrant BFS Candidate",
        source="qdrant",
        score=0.9,
        text="BFS shortest path",
        concepts=("BFS",),
        problem_type="Graph Traversal",
        payload={
            "source_id": "123",
            "answer": "Use BFS.",
            "solutionHints": ("Use a queue.",),
        },
    )
    understanding = QueryUnderstanding(
        original_query="BFS",
        normalized_query="BFS",
        input_kind="problem",
        intent="problem_search",
        keywords=("bfs",),
        concept_seeds=("BFS",),
    )
    trace = RetrievalTrace(
        query_understanding=understanding.to_mapping(),
        vector_candidates=[candidate.to_mapping()],
        fusion_scores=[candidate.to_mapping()],
        reranker_scores=[candidate.to_mapping()],
    )
    result = OnlineQueryResult(
        query_understanding=understanding,
        linked_entities=(),
        matched_problem=None,
        vector_candidates=(candidate,),
        graph_candidates=(),
        bm25_candidates=(),
        fused_candidates=(candidate,),
        reranked_candidates=(candidate,),
        graph_paths=(),
        trace=trace,
    )

    class StaticPipeline:
        def run(self, query: str, *, mode: str = "hybrid", top_k: int = 5) -> OnlineQueryResult:
            return result

    app.state.runtime_retrieval = RuntimeRetrieval(
        backend="stores",
        pipeline=StaticPipeline(),  # type: ignore[arg-type]
        candidate_sources={},
        provider_sources={},
    )

    response = TestClient(app).post("/api/analysis", json={"input": "BFS"})

    assert response.status_code == 200
    payload = response.json()
    evidence_problem = payload["evidenceBundle"]["similarProblems"][0]
    top_level_problem = payload["similarProblems"][0]
    assert (evidence_problem["source"], evidence_problem["sourceId"]) == ("qdrant", "123")
    assert (top_level_problem["source"], top_level_problem["sourceId"]) == ("qdrant", "123")


def test_filter_retrieval_trace_prefers_canonical_id_over_source_id_collision():
    trace = {
        "vectorCandidates": [
            {"id": "uva-437", "source": "UVa", "sourceId": "437"},
            {"id": "leetcode-437", "source": "UVa", "sourceId": "437"},
            {"source": "UVa", "sourceId": "437"},
        ]
    }
    evidence = {
        "matchedProblem": {"id": "uva-437", "source": "UVa", "sourceId": "437"},
        "similarProblems": [],
    }

    filtered = _filter_retrieval_trace(trace, evidence)

    assert [candidate.get("id") for candidate in filtered["vectorCandidates"]] == [
        "uva-437",
        None,
    ]


def test_filter_retrieval_trace_preserves_scoped_payload_diagnostics():
    matched_chunk = {
        "id": "uva-437:solution:0",
        "kind": "solution",
        "displayText": "Dynamic Programming display",
        "score": 0.9,
    }
    similar_chunk = {
        "id": "uva-10130:solution:0",
        "kind": "solution",
        "displayText": "0/1 Knapsack display",
        "score": 0.8,
    }
    trace = {
        "vectorCandidates": [
            {
                "id": "uva-437",
                "source": "vector",
                "concepts": ["DP", "LIS", "Sorting"],
                "problemType": "Dynamic Programming",
                "payload": {
                    "documentSource": "UVa",
                    "sourceId": "437",
                    "answer": "Safe DP answer",
                    "solutionHints": ["Safe DP hint"],
                    "promptContext": "Do not leak matched prompt context.",
                    "metadata": {
                        "source": "store",
                        "notes": "Do not leak matched notes.",
                    },
                    "rawChunks": [matched_chunk],
                    "chunkEvidence": [matched_chunk],
                    "rawChunksComplete": True,
                    "chunkCount": 1,
                    "concepts": ["DP", "LIS", "Sorting"],
                    "problemType": "Dynamic Programming",
                },
            },
            {
                "id": "uva-10130",
                "source": "vector",
                "concepts": ["DP", "0/1 Knapsack"],
                "problemType": "Dynamic Programming",
                "payload": {
                    "documentSource": "UVa",
                    "sourceId": "10130",
                    "answer": "0/1 Knapsack answer",
                    "solutionHints": ["Use 0/1 Knapsack"],
                    "rawAnswer": "Do not leak raw answer.",
                    "explanation": "Do not leak explanation.",
                    "rawChunks": [similar_chunk],
                    "chunkEvidence": [similar_chunk],
                    "rawChunksComplete": True,
                    "chunkCount": 1,
                    "concepts": ["DP", "0/1 Knapsack"],
                    "problemType": "Dynamic Programming",
                },
            },
        ]
    }
    evidence = {
        "matchedProblem": {
            "id": "uva-437",
            "source": "UVa",
            "sourceId": "437",
            "sharedConcepts": ["Dynamic Programming", "LIS", "Sorting"],
            "problemType": "Dynamic Programming",
        },
        "similarProblems": [
            {
                "id": "uva-10130",
                "source": "UVa",
                "sourceId": "10130",
                "sharedConcepts": ["Dynamic Programming"],
            }
        ],
        "algorithmEvidence": ["Dynamic Programming"],
    }

    filtered = _filter_retrieval_trace(trace, evidence)

    matched_payload = filtered["vectorCandidates"][0]["payload"]
    assert matched_payload["rawChunks"] == [matched_chunk]
    assert matched_payload["chunkEvidence"] == [matched_chunk]
    assert matched_payload["rawChunksComplete"] is True
    assert matched_payload["chunkCount"] == 1
    assert matched_payload["metadata"] == {"source": "store"}
    assert (matched_payload["documentSource"], matched_payload["sourceId"]) == ("UVa", "437")
    assert "answer" not in matched_payload
    assert "solutionHints" not in matched_payload
    assert "promptContext" not in matched_payload

    similar_payload = filtered["vectorCandidates"][1]["payload"]
    assert similar_payload["rawChunks"] == [
        {"id": "uva-10130:solution:0", "kind": "solution", "score": 0.8}
    ]
    assert similar_payload["chunkEvidence"] == [
        {"id": "uva-10130:solution:0", "kind": "solution", "score": 0.8}
    ]
    assert similar_payload["rawChunksComplete"] is True
    assert similar_payload["chunkCount"] == 1
    assert (similar_payload["documentSource"], similar_payload["sourceId"]) == (
        "UVa",
        "10130",
    )
    assert "answer" not in similar_payload
    assert "solutionHints" not in similar_payload
    assert "rawAnswer" not in similar_payload
    assert "explanation" not in similar_payload


def test_filter_retrieval_trace_sanitizes_nested_provenance_collections():
    trace = {
        "vectorCandidates": [
            {
                "id": "uva-437",
                "source": "vector",
                "concepts": ["DP"],
                "problemType": "Dynamic Programming",
                "payload": {
                    "documentSource": "UVa",
                    "sourceId": "437",
                    "provenance": [
                        "TRACE_SCALAR_PROVENANCE_POISON",
                        {
                            "source": "seed",
                            "sourceId": "437",
                            "notes": "TRACE_NESTED_PROVENANCE_POISON",
                            "metadata": {
                                "source": "seed-metadata",
                                "displayText": "TRACE_METADATA_POISON",
                            },
                        },
                    ],
                    "rawChunks": [
                        {
                            "id": "uva-437:solution:0",
                            "source": "vector",
                            "score": 0.9,
                            "payload": {
                                "kind": "solution",
                                "displayText": "Safe trace display text.",
                                "provenance": [
                                    "TRACE_RAW_CHUNK_SCALAR_POISON",
                                    {
                                        "source": "store",
                                        "sourceId": "437",
                                        "notes": "TRACE_RAW_CHUNK_NESTED_POISON",
                                        "metadata": {
                                            "source": "store-metadata",
                                            "displayText": "TRACE_RAW_CHUNK_METADATA_POISON",
                                        },
                                    },
                                ],
                            },
                        }
                    ],
                },
            }
        ]
    }
    evidence = {
        "matchedProblem": {
            "id": "uva-437",
            "source": "UVa",
            "sourceId": "437",
            "sharedConcepts": ["Dynamic Programming"],
            "problemType": "Dynamic Programming",
        },
        "similarProblems": [],
        "algorithmEvidence": ["Dynamic Programming"],
    }

    filtered = _filter_retrieval_trace(trace, evidence)
    payload = filtered["vectorCandidates"][0]["payload"]

    assert payload["provenance"] == [
        {
            "source": "seed",
            "sourceId": "437",
            "metadata": {"source": "seed-metadata"},
        }
    ]
    raw_chunk = payload["rawChunks"][0]
    assert raw_chunk["payload"]["provenance"] == [
        {
            "source": "store",
            "sourceId": "437",
            "metadata": {"source": "store-metadata"},
        }
    ]
    assert raw_chunk["payload"]["displayText"] == "Safe trace display text."
    assert "POISON" not in json.dumps(payload, ensure_ascii=False)


@pytest.mark.parametrize(
    ("scoped_problem_type", "expected"),
    [
        ("Dynamic Programming", "Dynamic Programming"),
        ("", ""),
    ],
)
def test_retrieval_problem_type_does_not_fall_back_to_raw_reranked_candidates(
    scoped_problem_type: str,
    expected: str,
):
    unrelated = RetrievalCandidate(
        id="leetcode-1091",
        title="Shortest Path in Binary Matrix",
        source="reranker",
        score=0.99,
        text="Use BFS.",
        concepts=("BFS",),
        problem_type="Graph Traversal",
    )

    class PipelineResult:
        matched_problem = None
        reranked_candidates = (unrelated,)

    evidence = {
        "patternEvidence": [],
        "similarProblems": [{"problemType": scoped_problem_type}],
    }

    assert _retrieval_problem_type(PipelineResult(), evidence) == expected


def test_problem_statement_returns_graph_traversal_bfs_analysis_contract():
    client = TestClient(app)

    response = client.post(
        "/api/analysis",
        json={
            "problemText": (
                "\u7d66\u5b9a\u4e00\u5f35\u7121\u6b0a\u5716\u8207\u8d77\u9ede\u7d42\u9ede\uff0c"
                "\u8acb\u627e\u51fa\u5f9e\u8d77\u9ede\u5230\u7d42\u9ede\u7684\u6700\u77ed\u6b65\u6578\u3002"
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "queryId",
        "status",
        "usedMockData",
        "inputKind",
        "problemType",
        "requiredConcepts",
        "similarProblems",
        "similarityReason",
        "solvingHints",
        "commonMistakes",
        "evidencePaths",
        "retrievalConfig",
        "retrievalTrace",
        "evidenceBundle",
    }
    assert payload["status"] == "ok"
    assert payload["usedMockData"] is False
    assert payload["inputKind"] == "problem"
    assert payload["problemType"] == "\u5716\u8ad6\u904d\u6b77\uff08Graph Traversal\uff09"
    assert "Graph Traversal" in payload["problemType"]
    assert "\u5716\u8ad6\u904d\u6b77" in payload["problemType"]

    concepts = {concept["name"]: concept for concept in payload["requiredConcepts"]}
    assert {"BFS", "Queue", "Visited Array"}.issubset(concepts)
    assert concepts["BFS"]["kind"] == "algorithm"
    assert concepts["Visited Array"]["kind"] == "technique"
    assert concepts["Queue"]["description"]

    assert any(problem["source"] == "UVa" for problem in payload["similarProblems"])
    assert any(problem["source"] == "LeetCode" for problem in payload["similarProblems"])
    recognized_concepts = set(concepts) | {
        "DFS",
        "Flood Fill",
        "Graph Traversal",
        "Shortest Path",
        "Unweighted Shortest Path",
    }
    for problem in payload["similarProblems"]:
        assert set(problem) == {
            "source",
            "id",
            "sourceId",
            "title",
            "reason",
            "sharedConcepts",
            "answerHint",
        }
        assert problem["id"].startswith(("leetcode-", "uva-"))
        assert problem["sourceId"]
        shared_concepts = set(problem["sharedConcepts"])
        assert shared_concepts
        assert shared_concepts & recognized_concepts
        assert problem["answerHint"]
    assert any(
        {"BFS", "Queue"}.issubset(set(problem["sharedConcepts"]))
        for problem in payload["similarProblems"]
    )

    assert "\u7121\u6b0a\u5716\u4e2d\u627e\u6700\u77ed\u6b65\u6578" in payload["similarityReason"]
    assert "BFS \u627e\u6700\u77ed\u6b65\u6578" in payload["similarityReason"]
    first_evidence_problem = payload["evidenceBundle"]["similarProblems"][0]
    assert {"BFS", "Queue"}.issubset(set(first_evidence_problem["sharedConcepts"]))
    assert payload["solvingHints"] == first_evidence_problem["solutionHints"]
    hints_text = "\n".join(payload["solvingHints"])
    assert "Queue" in hints_text
    assert any(
        expected in hints_text
        for expected in ("\u8d77\u9ede", "\u5ea7\u6a19", "\u8ddd\u96e2")
    )
    assert "\u5148\u5efa\u5716" not in hints_text
    assert any("visited" in mistake and "\u5fd8\u8a18\u6a19\u8a18" in mistake for mistake in payload["commonMistakes"])
    assert any("queue \u521d\u59cb\u5316\u932f\u8aa4" in mistake for mistake in payload["commonMistakes"])

    understanding = payload["retrievalTrace"]["queryUnderstanding"]
    assert understanding["queryLanguage"] == "zh-Hant"
    assert {"無權圖", "最短步數"}.issubset(set(understanding["keywords"]))
    assert {"BFS", "Queue", "Shortest Path", "Graph Traversal"}.issubset(
        set(understanding["conceptSeeds"])
    )
    assert "breadth first search" in understanding["queryVariants"]["bm25"]

    assert payload["retrievalTrace"]["bm25Candidates"]
    assert payload["retrievalTrace"]["graphCandidates"]
    assert payload["evidencePaths"]
    assert payload["evidenceBundle"]["graphPaths"]


@pytest.mark.parametrize("query", ["Visited Array", "拜訪陣列", "visited 陣列"])
def test_query_language_classifies_visited_array_aliases_as_technique(query):
    understanding = QueryUnderstandingService().understand(query)
    linked_entities = EntityLinkingService().link(understanding)
    matching_seeds = [
        seed
        for seed in linked_entities
        if seed.get("entityId") == "concept:visited-array"
    ]
    assert matching_seeds
    assert {seed.get("entityType") for seed in matching_seeds} == {"technique"}


def test_detect_graph_traversal_signals_supports_multilingual_shortest_path_terms():
    signals = set(
        detect_graph_traversal_signals("請用廣搜與佇列找出無權圖中從起點到終點的最短步數。")
    )

    assert {"BFS", "Queue", "Unweighted shortest path", "Graph"}.issubset(signals)


def test_python_code_with_queue_deque_and_visited_is_classified_as_python():
    client = TestClient(app)

    response = client.post(
        "/api/v1/analysis",
        json={
            "code": """
from collections import deque

def shortest_path(graph, start):
    visited = set([start])
    q = deque([(start, 0)])
    while q:
        node, dist = q.popleft()
        for nxt in graph[node]:
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, dist + 1))
"""
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["inputKind"] == "python"
    assert payload["problemType"] == "\u5716\u8ad6\u904d\u6b77\uff08Graph Traversal\uff09"
    assert {concept["name"] for concept in payload["requiredConcepts"]} >= {
        "BFS",
        "Queue",
        "Visited Array",
    }


def test_cpp_code_with_queue_and_visited_is_classified_as_cpp():
    client = TestClient(app)

    response = client.post(
        "/api/analysis",
        json={
            "code": """
#include <queue>
#include <vector>
using namespace std;

int main() {
    queue<int> q;
    vector<int> visited(10);
    q.push(0);
}
"""
        },
    )

    assert response.status_code == 200
    assert response.json()["inputKind"] == "cpp"


def test_analysis_response_includes_retrieval_model_config_contract():
    client = TestClient(app)

    response = client.post("/api/analysis", json={"input": "unweighted graph shortest path BFS"})

    assert response.status_code == 200
    config = response.json()["retrievalConfig"]
    assert config == {
        "embeddingModel": "BAAI/bge-m3",
        "rerankerModel": "BAAI/bge-reranker-v2-m3",
        "language": "zh-Hant",
        "embeddingProvider": {
            "provider": "mock",
            "model": "BAAI/bge-m3",
        },
        "rerankerProvider": {
            "provider": "mock",
            "model": "BAAI/bge-reranker-v2-m3",
        },
    }


def test_analysis_response_includes_trace_and_evidence_without_context_by_default():
    client = TestClient(app)

    response = client.post("/api/analysis", json={"input": "unweighted graph shortest path BFS"})

    assert response.status_code == 200
    payload = response.json()
    assert "retrievalTrace" in payload
    assert "evidenceBundle" in payload
    assert "contextPreview" not in payload
    assert "retrievalBackend" not in payload
    assert "candidateSources" not in payload["retrievalTrace"]
    for candidate_section in (
        "vectorCandidates",
        "graphCandidates",
        "bm25Candidates",
        "fusionScores",
        "rerankerScores",
    ):
        assert all(
            "candidateSource" not in candidate
            for candidate in payload["retrievalTrace"][candidate_section]
        )
    assert payload["retrievalTrace"]["queryUnderstanding"]["intent"] == "problem_search"
    assert payload["evidenceBundle"]["similarProblems"]


def test_analysis_debug_mode_includes_context_preview_and_retrieval_backend():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "unweighted graph shortest path BFS"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrievalBackend"] == "local"
    assert "contextPreview" in payload
    assert "查詢理解" in payload["contextPreview"]
    assert payload["retrievalTrace"]["candidateSources"] == {
        "vector": "local",
        "graph": "local",
        "bm25": "local",
    }
    assert payload["retrievalTrace"]["vectorCandidates"][0]["candidateSource"] == "local"
    assert payload["retrievalTrace"]["bm25Candidates"][0]["candidateSource"] == "local"
    assert payload["retrievalTrace"]["providerSources"] == {
        "embedding": {
            "provider": "mock",
            "model": "BAAI/bge-m3",
        },
        "reranker": {
            "provider": "mock",
            "model": "BAAI/bge-reranker-v2-m3",
        },
    }
    assert payload["retrievalConfig"]["embeddingProvider"] == {
        "provider": "mock",
        "model": "BAAI/bge-m3",
    }
    assert payload["retrievalConfig"]["rerankerProvider"] == {
        "provider": "mock",
        "model": "BAAI/bge-reranker-v2-m3",
    }


def test_debug_trace_contains_raw_graph_paths_but_evidence_uses_pruned_paths():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "BFS shortest path queue graph traversal"},
    )

    assert response.status_code == 200
    payload = response.json()
    raw_paths = payload["retrievalTrace"].get("rawGraphPaths")
    graph_paths = payload["evidenceBundle"].get("graphPaths")
    assert raw_paths
    assert graph_paths
    assert len(graph_paths) <= len(raw_paths)
    assert "contextPreview" in payload
    assert payload["commonMistakes"] == payload["evidenceBundle"]["commonMistakes"]


def test_non_debug_trace_does_not_expose_raw_graph_paths():
    client = TestClient(app)

    response = client.post(
        "/api/analysis",
        json={"input": "BFS shortest path queue graph traversal"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "rawGraphPaths" not in payload["retrievalTrace"]
    assert "contextPreview" not in payload


def test_analysis_debug_mode_uses_configured_runtime_retrieval():
    app.state.runtime_retrieval = RuntimeRetrieval(
        backend="stores",
        pipeline=OnlineQueryPipeline(
            documents=(
                RetrievalDocument(
                    id="fake-runtime-bfs",
                    source="FakeJudge",
                    source_id="runtime-1",
                    title="Fake Runtime BFS",
                    text="BFS shortest path with a queue in a graph.",
                    answer="Use a fake runtime queue answer.",
                    concepts=("BFS", "Queue"),
                    problem_type="Graph Traversal",
                ),
            ),
            embedding_provider=DeterministicMockEmbeddingProvider(dimension=8),
        ),
        candidate_sources={
            "vector": "fake_vector",
            "graph": "fake_graph",
            "bm25": "fake_bm25",
        },
        provider_sources={
            "embedding": {
                "provider": "fake_embedding",
                "model": "fake-embedding-model",
                "adapter": "qdrant",
            },
            "reranker": {
                "provider": "mock",
                "model": "BAAI/bge-reranker-v2-m3",
            },
        },
    )
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "BFS shortest path queue graph traversal"},
    )

    assert response.status_code == 200
    payload = response.json()
    trace = payload["retrievalTrace"]
    assert payload["retrievalBackend"] == "stores"
    assert trace["candidateSources"] == {
        "vector": "fake_vector",
        "graph": "fake_graph",
        "bm25": "fake_bm25",
    }
    assert trace["vectorCandidates"][0]["id"] == "fake-runtime-bfs"
    assert trace["graphCandidates"][0]["id"] == "fake-runtime-bfs"
    assert trace["bm25Candidates"][0]["id"] == "fake-runtime-bfs"
    assert trace["vectorCandidates"][0]["candidateSource"] == "fake_vector"
    assert trace["graphCandidates"][0]["candidateSource"] == "fake_graph"
    assert trace["bm25Candidates"][0]["candidateSource"] == "fake_bm25"
    assert trace["providerSources"] == {
        "embedding": {
            "provider": "fake_embedding",
            "model": "fake-embedding-model",
            "adapter": "qdrant",
        },
        "reranker": {
            "provider": "mock",
            "model": "BAAI/bge-reranker-v2-m3",
        },
    }
    assert payload["retrievalConfig"]["embeddingProvider"] == {
        "provider": "fake_embedding",
        "model": "fake-embedding-model",
        "adapter": "qdrant",
    }
    assert payload["retrievalConfig"]["rerankerProvider"] == {
        "provider": "mock",
        "model": "BAAI/bge-reranker-v2-m3",
    }
    assert payload["evidenceBundle"]["similarProblems"][0]["id"] == "fake-runtime-bfs"


def test_analysis_exposes_compatibility_warnings_only_in_debug_trace(monkeypatch):
    warning = {
        "adapter": "qdrant",
        "severity": "warning",
        "message": "Qdrant client 1.18.0 is outside the supported server 1.15.3 minor range.",
    }
    monkeypatch.setattr(
        app.state,
        "runtime_retrieval",
        RuntimeRetrieval(
            backend="stores",
            pipeline=OnlineQueryPipeline(
                documents=(
                    RetrievalDocument(
                        id="fake-runtime-bfs",
                        source="FakeJudge",
                        source_id="runtime-1",
                        title="Fake Runtime BFS",
                        text="BFS shortest path with a queue in a graph.",
                        answer="Use a fake runtime queue answer.",
                        concepts=("BFS", "Queue"),
                        problem_type="Graph Traversal",
                    ),
                ),
                embedding_provider=DeterministicMockEmbeddingProvider(dimension=8),
            ),
            candidate_sources={
                "vector": "fake_vector",
                "graph": "fake_graph",
                "bm25": "fake_bm25",
            },
            provider_sources={},
            compatibility_warnings=[warning],
        ),
        raising=False,
    )
    client = TestClient(app)

    debug_response = client.post(
        "/api/analysis?debug=true",
        json={"input": "BFS shortest path queue graph traversal"},
    )
    normal_response = client.post(
        "/api/analysis",
        json={"input": "BFS shortest path queue graph traversal"},
    )

    assert debug_response.status_code == 200
    assert debug_response.json()["retrievalTrace"]["compatibilityWarnings"] == [warning]
    assert normal_response.status_code == 200
    assert "compatibilityWarnings" not in normal_response.json()["retrievalTrace"]


def test_analysis_uses_only_canonical_graph_paths_for_top_level_evidence():
    candidate = RetrievalCandidate(
        id="fake-runtime-bfs",
        title="Fake Runtime BFS",
        source="vector",
        score=0.91,
        text="BFS shortest path with a queue in a graph.",
        concepts=("BFS", "Queue"),
        problem_type="Graph Traversal",
        payload={
            "documentSource": "FakeJudge",
            "sourceId": "runtime-1",
            "answer": "Use a fake runtime queue answer.",
        },
    )
    expected_result = OnlineQueryResult(
        query_understanding=QueryUnderstanding(
            original_query="BFS shortest path with queue",
            normalized_query="BFS shortest path with queue",
            input_kind="problem",
            intent="problem_search",
            keywords=("bfs", "shortest", "path", "queue"),
        ),
        linked_entities=(),
        matched_problem=None,
        vector_candidates=(candidate,),
        graph_candidates=(),
        bm25_candidates=(),
        fused_candidates=(candidate,),
        reranked_candidates=(candidate,),
        graph_paths=(),
        trace=RetrievalTrace(
            query_understanding={
                "originalQuery": "BFS shortest path with queue",
                "normalizedQuery": "BFS shortest path with queue",
                "inputKind": "problem",
                "intent": "problem_search",
                "keywords": ["bfs", "shortest", "path", "queue"],
            },
            vector_candidates=[candidate.to_mapping()],
            fusion_scores=[candidate.to_mapping()],
            reranker_scores=[candidate.to_mapping()],
        ),
    )

    class FusedCandidateNoGraphPathPipeline:
        def run(
            self,
            query: str,
            *,
            mode: str = "hybrid",
            top_k: int = 5,
        ) -> OnlineQueryResult:
            assert query == "BFS shortest path with queue"
            assert mode == "hybrid"
            assert top_k == 5
            return expected_result

    app.state.runtime_retrieval = RuntimeRetrieval(
        backend="stores",
        pipeline=FusedCandidateNoGraphPathPipeline(),  # type: ignore[arg-type]
        candidate_sources={
            "vector": "fake_vector",
            "graph": "fake_graph",
            "bm25": "fake_bm25",
        },
        provider_sources={},
    )
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "BFS shortest path with queue"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["evidenceBundle"]["similarProblems"][0]["id"] == "fake-runtime-bfs"
    assert payload["evidenceBundle"]["graphPaths"] == []
    assert payload["evidencePaths"] == []


def test_analysis_paths_skip_non_mapping_items_and_convert_later_valid_paths():
    converted = _analysis_paths_from_graph_trace(
        [
            None,
            {
                "nodes": ["input", "uva-10653"],
                "relations": ["EXACT_MATCH"],
                "score": 1.0,
                "pathSource": "neo4j",
            },
        ]
    )

    assert len(converted) == 1
    assert converted[0].title == "Graph path 1 (neo4j)"
    assert converted[0].edges[0].weight == 1.0


def test_analysis_paths_skip_inconsistent_node_relation_counts():
    converted = _analysis_paths_from_graph_trace(
        [
            {
                "nodes": ["input", "uva-10653", "concept:bfs"],
                "relations": ["EXACT_MATCH"],
                "score": 1.0,
                "pathSource": "neo4j",
            },
            {
                "nodes": ["input", "uva-10653"],
                "relations": ["EXACT_MATCH"],
                "score": 1.0,
                "pathSource": "inferred",
            },
        ]
    )

    assert len(converted) == 1
    assert converted[0].title == "Graph path 1 (inferred)"
    assert [node.id for node in converted[0].nodes] == ["input", "uva-10653"]


@pytest.mark.parametrize("score", ["bad", None, float("nan"), float("inf"), float("-inf")])
def test_analysis_paths_use_zero_score_for_malformed_or_non_finite_values(score):
    converted = _analysis_paths_from_graph_trace(
        [
            {
                "nodes": ["input", "concept:bfs"],
                "relations": ["REQUIRES"],
                "score": score,
            }
        ]
    )

    assert converted[0].edges[0].weight == 0.0


def test_analysis_graph_mode_empty_reranked_candidates_keeps_top_level_similar_empty():
    matched_candidate = RetrievalCandidate(
        id="uva-10653",
        title="Bombs! NO they are Mines!!",
        source="UVa",
        score=1.0,
        text="Find the shortest safe path on a grid with bomb cells.",
        concepts=("BFS", "Queue", "Visited Array"),
        problem_type="Graph Traversal",
        payload={
            "documentSource": "UVa",
            "sourceId": "10653",
            "answer": "Run BFS from the start cell while skipping bomb cells.",
            "solutionHints": ("Mark bomb cells before BFS.",),
            "difficulty": "Medium",
            "constraints": (),
        },
    )
    matched_problem = ExactProblemMatch(
        problem_id=matched_candidate.id,
        title=matched_candidate.title,
        source=matched_candidate.source,
        source_id="10653",
        match_kind="exact_title",
        confidence=1.0,
        candidate=matched_candidate,
    )
    expected_result = OnlineQueryResult(
        query_understanding=QueryUnderstanding(
            original_query="UVA-10653 - Bombs! NO they are Mines!!",
            normalized_query="uva-10653 bombs no they are mines",
            input_kind="problem",
            intent="problem_search",
            keywords=("uva", "10653", "bombs", "mines"),
        ),
        linked_entities=(),
        matched_problem=matched_problem,
        vector_candidates=(),
        graph_candidates=(),
        bm25_candidates=(),
        fused_candidates=(),
        reranked_candidates=(),
        graph_paths=(
            {
                "nodes": ["input", "uva-10653", "concept:bfs"],
                "relations": ["EXACT_MATCH", "REQUIRES"],
                "score": 1.0,
                "pathSource": "neo4j",
            },
        ),
        trace=RetrievalTrace(
            query_understanding={
                "originalQuery": "UVA-10653 - Bombs! NO they are Mines!!",
                "normalizedQuery": "uva-10653 bombs no they are mines",
                "inputKind": "problem",
                "intent": "problem_search",
                "keywords": ["uva", "10653", "bombs", "mines"],
            },
            matched_problem=matched_problem.to_mapping(),
        ),
    )

    class EmptyGraphPipeline:
        def run(
            self,
            query: str,
            *,
            mode: str = "hybrid",
            top_k: int = 5,
        ) -> OnlineQueryResult:
            assert query == "UVA-10653 - Bombs! NO they are Mines!!"
            assert mode == "graph"
            assert top_k == 3
            return expected_result

    app.state.runtime_retrieval = RuntimeRetrieval(
        backend="stores",
        pipeline=EmptyGraphPipeline(),  # type: ignore[arg-type]
        candidate_sources={},
        provider_sources={},
    )
    client = TestClient(app)

    response = client.post(
        "/api/analysis",
        json={
            "input": "UVA-10653 - Bombs! NO they are Mines!!",
            "mode": "graph",
            "topK": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["similarProblems"] == []
    assert payload["evidenceBundle"]["similarProblems"] == []
    assert payload["matchedProblem"]["id"] == "uva-10653"
    assert payload["evidenceBundle"]["matchedProblem"]["id"] == "uva-10653"
    assert payload["evidencePaths"][0]["nodes"][1]["id"] == "uva-10653"
    assert payload["requiredConcepts"]
    assert payload["solvingHints"]
    assert payload["commonMistakes"]


def test_analysis_problem_id_only_preserves_exact_id_retrieval():
    matched_candidate = RetrievalCandidate(
        id="uva-10653",
        title="Bombs! NO they are Mines!!",
        source="UVa",
        score=1.0,
        text="Find the shortest safe path on a grid with bomb cells.",
        concepts=("BFS", "Queue", "Visited Array"),
        problem_type="Graph Traversal",
        payload={
            "documentSource": "UVa",
            "sourceId": "10653",
            "answer": "Run BFS from the start cell while skipping bomb cells.",
            "solutionHints": ("Mark bomb cells before BFS.",),
            "difficulty": "Medium",
            "constraints": (),
        },
    )
    similar_candidate = RetrievalCandidate(
        id="leetcode-1091",
        title="Shortest Path in Binary Matrix",
        source="LeetCode",
        score=0.82,
        text="Use BFS to find a shortest path in an unweighted binary matrix.",
        concepts=("BFS", "Queue", "Visited Array"),
        problem_type="Graph Traversal",
        payload={
            "documentSource": "LeetCode",
            "sourceId": "1091",
            "answer": "Run BFS over eight directions.",
            "solutionHints": ("Use a queue.",),
            "difficulty": "Medium",
            "constraints": (),
        },
    )
    no_concept_similar_candidate = RetrievalCandidate(
        id="leetcode-2000",
        title="Grid Path Without Concepts",
        source="LeetCode",
        score=0.72,
        text="Find a path through an unweighted grid.",
        concepts=(),
        problem_type="Graph Traversal",
        payload={
            "documentSource": "LeetCode",
            "sourceId": "2000",
            "answer": "Traverse the grid from the start cell.",
            "solutionHints": ("Use the graph structure.",),
            "difficulty": "Easy",
            "constraints": (),
        },
    )
    matched_problem = ExactProblemMatch(
        problem_id=matched_candidate.id,
        title=matched_candidate.title,
        source=matched_candidate.source,
        source_id="10653",
        match_kind="exact_problem_id",
        confidence=1.0,
        candidate=matched_candidate,
    )
    trace = RetrievalTrace(
        query_understanding={
            "originalQuery": "uva-10653",
            "normalizedQuery": "uva-10653",
            "inputKind": "problem",
            "intent": "problem_search",
            "keywords": ["uva", "10653"],
        },
        vector_candidates=[matched_candidate.to_mapping()],
        graph_candidates=[matched_candidate.to_mapping()],
        bm25_candidates=[matched_candidate.to_mapping()],
        fusion_scores=[matched_candidate.to_mapping()],
        reranker_scores=[matched_candidate.to_mapping()],
        matched_problem=matched_problem.to_mapping(),
    )
    expected_result = OnlineQueryResult(
        query_understanding=QueryUnderstanding(
            original_query="uva-10653",
            normalized_query="uva-10653",
            input_kind="problem",
            intent="problem_search",
            keywords=("uva", "10653"),
        ),
        linked_entities=(),
        matched_problem=matched_problem,
        vector_candidates=(matched_candidate,),
        graph_candidates=(matched_candidate,),
        bm25_candidates=(matched_candidate,),
        fused_candidates=(matched_candidate,),
        reranked_candidates=(
            matched_candidate,
            similar_candidate,
            no_concept_similar_candidate,
        ),
        graph_paths=(
            {
                "nodes": [],
                "relations": [],
                "score": 1.0,
                "pathSource": "neo4j",
            },
            {
                "nodes": ["input", "uva-10653", "concept:bfs"],
                "relations": ["EXACT_MATCH", "REQUIRES"],
                "score": 1.0,
                "pathSource": "neo4j",
                "rationale": "Neo4j returned the exact problem path.",
                "storePath": {
                    "nodes": ["uva-10653", "concept:bfs"],
                    "relations": ["REQUIRES"],
                },
            },
        ),
        trace=trace,
    )

    class RecordingPipeline:
        def __init__(self, result: OnlineQueryResult) -> None:
            self.result = result
            self.last_query: str | None = None
            self.last_mode: str | None = None
            self.last_top_k: int | None = None

        def run(
            self,
            query: str,
            *,
            mode: str = "hybrid",
            top_k: int = 5,
        ) -> OnlineQueryResult:
            self.last_query = query
            self.last_mode = mode
            self.last_top_k = top_k
            return self.result

    pipeline = RecordingPipeline(expected_result)
    app.state.runtime_retrieval = RuntimeRetrieval(
        backend="stores",
        pipeline=pipeline,  # type: ignore[arg-type]
        candidate_sources={
            "vector": "fake_vector",
            "graph": "fake_graph",
            "bm25": "fake_bm25",
        },
        provider_sources={},
    )
    client = TestClient(app)

    response = client.post(
        "/api/analysis",
        json={
            "problemId": "uva-10653",
            "mode": "graph",
            "topK": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert pipeline.last_query == "uva-10653"
    assert pipeline.last_mode == "graph"
    assert pipeline.last_top_k == 3
    assert payload["matchedProblem"]["id"] == "uva-10653"
    assert payload["retrievalTrace"]["matchedProblem"]["id"] == "uva-10653"
    assert payload["evidenceBundle"]["matchedProblem"]["id"] == "uva-10653"
    assert all(problem["id"] != "uva-10653" for problem in payload["evidenceBundle"]["similarProblems"])
    assert [problem["id"] for problem in payload["similarProblems"]] == [
        "leetcode-1091",
        "leetcode-2000",
    ]
    assert [problem["sourceId"] for problem in payload["similarProblems"]] == ["1091", "2000"]
    assert payload["similarProblems"][0]["source"] == "LeetCode"
    assert payload["similarProblems"][0]["title"] == "Shortest Path in Binary Matrix"
    assert payload["similarProblems"][0]["reason"] == (
        "所選檢索模式將此題列為最終候選，並共享這些概念：BFS、Queue、Visited Array。"
    )
    assert payload["similarProblems"][1]["reason"] == "所選檢索模式將此題列為最終重排序候選。"
    assert payload["evidencePaths"] == [
        {
            "title": "Graph path 1 (neo4j)",
            "nodes": [
                {"id": "input", "label": "input", "type": "input"},
                {"id": "uva-10653", "label": "uva-10653", "type": "problem"},
                {"id": "concept:bfs", "label": "concept:bfs", "type": "concept"},
            ],
            "edges": [
                {
                    "from": "input",
                    "to": "uva-10653",
                    "relation": "EXACT_MATCH",
                    "weight": 1.0,
                },
                {
                    "from": "uva-10653",
                    "to": "concept:bfs",
                    "relation": "REQUIRES",
                    "weight": 1.0,
                },
            ],
        }
    ]


@pytest.mark.parametrize("field_name", ["topK", "top_k"])
def test_analysis_request_accepts_top_k_aliases(field_name):
    request = AnalysisRequest.model_validate(
        {
            "input": "BFS shortest path",
            "mode": "vector",
            field_name: 8,
        }
    )

    assert request.mode == "vector"
    assert request.top_k == 8


@pytest.mark.parametrize(
    ("payload", "error_location"),
    [
        ({"input": "BFS", "mode": "invalid"}, ("mode",)),
        ({"input": "BFS", "topK": 0}, ("topK",)),
        ({"input": "BFS", "top_k": 11}, ("top_k",)),
    ],
)
def test_analysis_request_rejects_invalid_mode_and_top_k(payload, error_location):
    with pytest.raises(ValueError) as exc_info:
        AnalysisRequest.model_validate(payload)

    assert error_location in {
        tuple(error["loc"])
        for error in exc_info.value.errors()
    }


def test_analysis_exact_problem_query_exposes_consistent_matched_problem():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "UVA-10653 - Bombs! NO they are Mines!!", "topK": 10},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["inputKind"] == "problem"
    assert payload["queryId"].startswith("analysis-problem-")
    assert payload["matchedProblem"]["id"] == "uva-10653"
    assert (
        payload["retrievalTrace"]["matchedProblem"]["id"]
        == payload["matchedProblem"]["id"]
        == payload["evidenceBundle"]["matchedProblem"]["id"]
    )
    assert all(
        problem["id"] != "uva-10653"
        for problem in payload["evidenceBundle"]["similarProblems"]
    )
    top_level_similar_problems = payload["similarProblems"]
    assert top_level_similar_problems
    top_level_similar_problem_ids = {
        problem["id"] for problem in top_level_similar_problems
    }
    top_level_source_ids = {
        str(problem["sourceId"]) for problem in top_level_similar_problems
    }
    assert all(
        problem_id.startswith(("leetcode-", "uva-"))
        for problem_id in top_level_similar_problem_ids
    )
    assert all(problem["sourceId"] for problem in top_level_similar_problems)
    assert {"uva-10653", "10653"}.isdisjoint(
        top_level_similar_problem_ids | top_level_source_ids
    )


def test_analysis_exact_sliding_window_problem_uses_retrieval_fields():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "uva-1121 Subsequence", "mode": "hybrid", "topK": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["matchedProblem"]["id"] == "uva-1121"
    assert payload["matchedProblem"]["problemType"] == "Sliding Window"
    assert payload["problemType"] == "Sliding Window"

    concept_names = _required_concept_names(payload)
    assert {"Sliding Window", "Two Pointers"}.issubset(concept_names)
    assert "BFS" not in concept_names

    hint_text = " ".join(payload["solvingHints"]).lower()
    assert "left" in hint_text or "right" in hint_text or "指標" in hint_text
    assert "先建圖" not in hint_text
    assert "uva-1121" not in _similar_problem_ids(payload)


def test_analysis_exact_problem_reports_paths_only_graph_status():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "uva-1121 Subsequence", "mode": "hybrid", "topK": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    trace = payload["retrievalTrace"]
    assert trace["graphCandidates"] == []
    assert payload["evidenceBundle"]["graphPaths"]
    assert trace["graphSearchStatus"] == "paths_only"


def test_analysis_uses_canonical_problem_ids_across_response_surfaces():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={
            "input": "UVA-10653 - Bombs! NO they are Mines!!",
            "mode": "hybrid",
            "topK": 10,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    top_level_similar_problems = payload["similarProblems"]
    canonical_prefixes = ("leetcode-", "uva-")

    def canonical_ids(records, *, id_key: str):
        ids = {str(record[id_key]) for record in records if record.get(id_key)}
        assert ids
        assert all(problem_id.startswith(canonical_prefixes) for problem_id in ids)
        return ids

    top_level_ids = canonical_ids(top_level_similar_problems, id_key="id")

    assert all("sourceId" in problem for problem in top_level_similar_problems)
    top_level_id_source_id_pairs = _id_source_pairs(top_level_similar_problems)

    evidence_similar_problems = payload["evidenceBundle"]["similarProblems"]
    evidence_ids = canonical_ids(evidence_similar_problems, id_key="id")
    evidence_id_source_id_pairs = _id_source_pairs(
        payload["evidenceBundle"]["similarProblems"]
    )
    reranker_ids = canonical_ids(
        payload["retrievalTrace"]["rerankerScores"],
        id_key="id",
    )
    assert top_level_id_source_id_pairs == evidence_id_source_id_pairs
    assert {"uva-10653", "10653"}.isdisjoint(
        {
            value
            for pair in top_level_id_source_id_pairs
            for value in pair
        }
    )
    assert top_level_ids < reranker_ids
    assert payload["matchedProblem"]["id"] == "uva-10653"
    assert "uva-10653" not in evidence_ids
    assert "uva-10653" in reranker_ids


def test_analysis_exact_source_id_vector_query_preserves_retrieval_similar_problems():
    client = TestClient(app)

    response = client.post(
        "/api/analysis",
        json={"input": "10653", "mode": "vector", "topK": 10},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matchedProblem"]["id"] == "uva-10653"
    assert payload["matchedProblem"]["sourceId"] == "10653"
    top_level_similar_problems = payload["similarProblems"]
    assert top_level_similar_problems
    assert all(
        problem["id"].startswith(("leetcode-", "uva-"))
        for problem in top_level_similar_problems
    )
    assert all(problem["sourceId"] for problem in top_level_similar_problems)
    assert all(problem["title"] for problem in top_level_similar_problems)
    assert all(problem["sharedConcepts"] for problem in top_level_similar_problems)
    assert all(
        problem["source"] in {"LeetCode", "UVa"}
        for problem in top_level_similar_problems
    )
    assert {"uva-10653", "10653"}.isdisjoint(
        {
            value
            for problem in top_level_similar_problems
            for value in (problem["id"], problem["sourceId"])
        }
    )
    assert all(problem["reason"] for problem in top_level_similar_problems)
    assert all(
        "Retrieved by selected mode" not in problem["reason"]
        for problem in top_level_similar_problems
    )
    assert all(
        any(concept in problem["reason"] for concept in problem["sharedConcepts"])
        for problem in top_level_similar_problems
    )


def test_analysis_unknown_explicit_problem_id_returns_404():
    client = TestClient(app)

    response = client.post("/api/analysis", json={"problemId": "leetcode-404"})

    assert response.status_code == 404


@pytest.mark.parametrize(
    "input_text",
    [
        "I ate rotting oranges for breakfast.",
        "I ate oranges for dinner and want weather",
    ],
)
def test_analysis_unknown_unrelated_input_abstains_without_graph_or_bfs_evidence(input_text: str):
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={
            "input": input_text,
            "mode": "hybrid",
            "topK": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unsupported"
    assert payload["inputKind"] == "unknown"
    assert payload["problemType"] == ""
    assert payload["similarityReason"] == ""
    assert (
        payload["abstentionReason"]
        == "未偵測到程式題、程式碼、演算法概念或可靠檢索證據。"
    )
    assert payload["requiredConcepts"] == []
    assert payload["similarProblems"] == []
    assert payload["solvingHints"] == []
    assert payload["commonMistakes"] == []
    assert payload["evidencePaths"] == []
    assert payload["evidenceBundle"]["graphPaths"] == []
    assert payload["evidenceBundle"]["similarProblems"] == []
    assert payload["evidenceBundle"]["algorithmEvidence"] == []
    assert payload["evidenceBundle"]["dataStructureEvidence"] == []
    assert payload["evidenceBundle"]["patternEvidence"] == []
    assert payload["evidenceBundle"]["techniqueEvidence"] == []
    assert payload["evidenceBundle"]["commonMistakes"] == []
    assert payload["matchedProblem"] is None
    assert payload["retrievalTrace"]["entityLinking"] == []
    assert payload["retrievalTrace"]["vectorCandidates"] == []
    assert payload["retrievalTrace"]["graphCandidates"] == []
    assert payload["retrievalTrace"]["bm25Candidates"] == []
    assert payload["retrievalTrace"]["fusionScores"] == []
    assert payload["retrievalTrace"]["rerankerScores"] == []
    assert "BFS" not in response.text


def test_analysis_bare_queue_concept_query_returns_related_graph_problems():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "queue", "mode": "hybrid", "topK": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload.get("abstentionReason") is None

    understanding = payload["retrievalTrace"]["queryUnderstanding"]
    assert "Queue" in understanding["conceptSeeds"]
    assert "concept:queue" in understanding["queryVariants"]["graphSeeds"]

    graph_candidate_ids = {
        candidate["id"]
        for candidate in payload["retrievalTrace"]["graphCandidates"]
    }
    assert graph_candidate_ids & {
        "leetcode-1091",
        "leetcode-994",
        "uva-10653",
        "uva-11624",
        "uva-532",
    }
    assert "Queue" in _required_concept_names(payload)
    assert payload["similarProblems"]


@pytest.mark.parametrize(
    "input_text",
    [
        "queue lunch",
        "deque dinner",
        "I saw a queue at lunch",
        "the node at the edge of the restaurant was busy",
    ],
)
def test_analysis_weak_traversal_vocabulary_still_abstains(input_text: str):
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={
            "input": input_text,
            "mode": "hybrid",
            "topK": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unsupported"
    assert payload["inputKind"] == "unknown"
    assert payload["requiredConcepts"] == []
    assert payload["similarProblems"] == []
    assert payload["evidencePaths"] == []
    assert payload["evidenceBundle"]["graphPaths"] == []
    assert (
        payload["abstentionReason"]
        == "輸入超出目前支援範圍，請提供程式題敘、題號、程式碼或已知演算法概念。"
    )
    assert payload["matchedProblem"] is None
    assert "BFS" not in response.text


@pytest.mark.parametrize("query", ["DP", "dynamic programming", "動態規劃"])
def test_analysis_dynamic_programming_concept_query_returns_dp_candidates(query: str):
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": query, "mode": "hybrid", "topK": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload.get("abstentionReason") is None

    understanding = payload["retrievalTrace"]["queryUnderstanding"]
    assert "Dynamic Programming" in understanding["conceptSeeds"]
    assert "concept:dynamic-programming" in understanding["queryVariants"]["graphSeeds"]

    concept_names = _required_concept_names(payload)
    assert "Dynamic Programming" in concept_names
    assert {"BFS", "Queue"}.isdisjoint(concept_names)

    top_level_pairs = _id_source_pairs(payload["similarProblems"])
    evidence_pairs = _id_source_pairs(payload["evidenceBundle"]["similarProblems"])
    assert top_level_pairs == evidence_pairs
    assert ("uva-437", "437") in top_level_pairs
    assert all(problem_id != "leetcode-1091" for problem_id, _ in top_level_pairs)

    text = _response_text(payload)
    assert "Shortest Path in Binary Matrix" not in text
    assert "Rotting Oranges" not in text


def test_analysis_uva_437_top_level_similar_problems_match_filtered_evidence_pairs():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "uva-437", "mode": "hybrid", "topK": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matchedProblem"]["id"] == "uva-437"
    assert _id_source_pairs(payload["similarProblems"]) == _id_source_pairs(
        payload["evidenceBundle"]["similarProblems"]
    )
    assert {
        ("uva-437", "437"),
    }.isdisjoint(_id_source_pairs(payload["similarProblems"]))


@pytest.mark.parametrize("top_k", [5, 10])
def test_analysis_uva_437_exact_match_does_not_include_bfs_evidence_or_required_concepts(top_k: int):
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "uva-437", "mode": "hybrid", "topK": top_k},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["matchedProblem"]["id"] == "uva-437"
    assert payload["matchedProblem"]["sourceId"] == "437"

    concept_names = _required_concept_names(payload)
    matched_scope = {"Dynamic Programming", "LIS", "Sorting"}
    assert concept_names <= matched_scope
    assert "Dynamic Programming" in concept_names
    assert {
        "BFS",
        "Queue",
        "Visited Array",
        "0/1 Knapsack",
        "Hash Map",
        "Frequency Counting",
        "Binary Search",
        "Stack",
    }.isdisjoint(concept_names)

    evidence = payload["evidenceBundle"]
    assert "Dynamic Programming" in evidence["algorithmEvidence"]
    assert "BFS" not in evidence["algorithmEvidence"]
    assert "Queue" not in evidence["dataStructureEvidence"]
    assert "Graph Traversal" not in evidence["patternEvidence"]

    blocked_ids = {"leetcode-1091", "leetcode-994"}
    top_level_pairs = _id_source_pairs(payload["similarProblems"])
    evidence_pairs = _id_source_pairs(evidence["similarProblems"])
    assert top_level_pairs == evidence_pairs
    assert blocked_ids.isdisjoint({problem_id for problem_id, _ in top_level_pairs})
    assert all(
        set(problem["sharedConcepts"]) <= matched_scope
        for problem in evidence["similarProblems"]
    )

    text = _response_text(payload)
    assert "Shortest Path in Binary Matrix" not in text
    assert "Rotting Oranges" not in text
    assert "Fire!" not in text

    scoped_surface_text = _response_text(
        {
            "topLevelAnswerHints": [
                problem["answerHint"] for problem in payload["similarProblems"]
            ],
            "solvingHints": payload["solvingHints"],
            "similarProblems": payload["similarProblems"],
            "evidenceBundle": payload["evidenceBundle"],
            "contextPreview": payload["contextPreview"],
            "retrievalTrace": payload["retrievalTrace"],
        }
    )
    for unrelated_text in (
        "BFS",
        "Queue",
        "Visited Array",
        "Graph Traversal",
    ):
        assert unrelated_text not in scoped_surface_text


def test_analysis_mixed_bfs_uva_437_query_keeps_tower_scope_across_public_evidence():
    response = TestClient(app).post(
        "/api/analysis?debug=true",
        json={"input": "BFS uva-437", "mode": "hybrid", "topK": 10},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matchedProblem"]["id"] == "uva-437"
    assert "Dynamic Programming" in _required_concept_names(payload)
    assert payload["evidenceBundle"]["graphPaths"]

    public_evidence = {
        "problemType": payload["problemType"],
        "requiredConcepts": payload["requiredConcepts"],
        "similarProblems": payload["similarProblems"],
        "solvingHints": payload["solvingHints"],
        "commonMistakes": payload["commonMistakes"],
        "evidencePaths": payload["evidencePaths"],
        "evidenceBundle": payload["evidenceBundle"],
        "contextPreview": payload["contextPreview"],
    }
    public_text = json.dumps(public_evidence, ensure_ascii=False)
    for foreign_term in (
        "BFS",
        "Queue",
        "Visited Array",
        "Graph Traversal",
        "Hash Map",
        "Frequency Counting",
        "0/1 Knapsack",
        "Binary Search",
        "Stack",
        "Hardwood Species",
    ):
        assert foreign_term not in public_text


def test_analysis_partial_title_match_is_diagnostic_only():
    response = TestClient(app).post(
        "/api/analysis?debug=true",
        json={"input": "BFS shortest path", "mode": "hybrid", "topK": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload.get("matchedProblem") is None
    assert payload["evidenceBundle"]["matchedProblem"] is None
    assert payload["retrievalTrace"]["matchedProblem"]["matchKind"] == "partial_title"
    assert "直接命中" not in payload["similarityReason"]


def test_analysis_sample_input_uses_one_coherent_shortest_path_hint_set():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={
            "input": "給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。需要說明該使用哪些演算法與資料結構。"
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert {"BFS", "Queue"}.issubset(_required_concept_names(payload))
    assert payload["solvingHints"]
    assert (
        payload["solvingHints"]
        == payload["evidenceBundle"]["similarProblems"][0]["solutionHints"]
    )

    hints_text = "\n".join(payload["solvingHints"])
    assert "Queue" in hints_text
    assert any(
        expected in hints_text
        for expected in ("先檢查起點", "座標與距離")
    )
    assert "先建圖" not in hints_text
    assert not any(
        unrelated in hints_text
        for unrelated in (
            "火",
            "fire_time",
            "油田",
            "DFS",
            "腐爛",
            "Rotting",
            "Fire!",
            "Oil Deposits",
        )
    )


def test_analysis_scope_abstention_reason_is_zh_hant():
    response = TestClient(app).post(
        "/api/analysis?debug=true",
        json={"input": "queue lunch", "mode": "hybrid", "topK": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unsupported"
    assert (
        payload["abstentionReason"]
        == "輸入超出目前支援範圍，請提供程式題敘、題號、程式碼或已知演算法概念。"
    )


def test_analysis_supported_framed_problem_reference_keeps_supported_fields():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "please analyze UVA-10653 - Bombs! NO they are Mines!!"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["matchedProblem"]["id"] == "uva-10653"
    assert payload["requiredConcepts"]
    assert payload["problemType"] == "圖論遍歷（Graph Traversal）"
    assert payload["problemType"] != "不支援的問題"


@pytest.mark.parametrize(
    "input_text",
    ["please analyze UVA-10653", "please analyze 10653"],
)
def test_analysis_wrapped_known_problem_id_keeps_supported_fields(input_text: str):
    client = TestClient(app)

    response = client.post("/api/analysis", json={"input": input_text})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["inputKind"] == "problem"
    assert payload["requiredConcepts"]
    assert payload["problemType"] != "不支援的問題"


@pytest.mark.parametrize("route", ["/api/analysis", "/api/v1/analysis"])
def test_analysis_rejects_oversized_input_with_structured_limit_response(route: str):
    client = TestClient(app)
    oversized = " " * 9001

    response = client.post(f"{route}?debug=true", json={"input": oversized})

    assert response.status_code == 413
    assert response.json()["detail"] == {
        "code": "input_too_large",
        "maxLength": 8000,
        "actualLength": 9001,
    }


def test_analysis_openapi_documents_input_limit_response():
    client = TestClient(app)

    openapi = client.get("/openapi.json").json()

    assert "413" in openapi["paths"]["/api/analysis"]["post"]["responses"]
    assert "413" in openapi["paths"]["/api/v1/analysis"]["post"]["responses"]


def test_dataset_loader_contains_clean_traditional_chinese_answers_and_hints():
    dataset = load_programming_dataset()
    examples = find_graph_traversal_examples(dataset)

    assert examples
    assert any(problem.source == "UVa" for problem in examples)
    assert any(problem.source == "LeetCode" for problem in examples)
    assert all(problem.answer for problem in examples)
    assert all(problem.solution_hints for problem in examples)
    assert any("\u6700\u77ed\u6b65\u6578" in problem.answer for problem in examples)
    assert all("\ufffd" not in problem.answer for problem in examples)
    assert all("?" not in problem.answer for problem in examples)

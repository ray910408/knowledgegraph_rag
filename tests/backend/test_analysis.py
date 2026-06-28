from collections.abc import Iterator

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
    _similar_problem_responses_from_candidates,
    app,
)
from backend.app.providers import DeterministicMockEmbeddingProvider
from backend.app.retrieval.pipeline import (
    ExactProblemMatch,
    OnlineQueryPipeline,
    OnlineQueryResult,
    QueryUnderstanding,
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


def test_similar_problem_responses_keep_source_ids_optional_and_accept_aliases():
    def candidate(candidate_id: str, payload: dict[str, str]) -> RetrievalCandidate:
        return RetrievalCandidate(
            id=candidate_id,
            title=candidate_id,
            source="LeetCode",
            score=0.8,
            text="BFS shortest path",
            concepts=("BFS",),
            problem_type="Graph Traversal",
            payload=payload,
        )

    responses = _similar_problem_responses_from_candidates(
        (
            candidate("leetcode-matched", {}),
            candidate("leetcode-source-matched", {"sourceId": "matched-source"}),
            candidate("leetcode-no-source-id", {}),
            candidate("leetcode-snake-case-source", {"source_id": "snake-source"}),
        ),
        matched_problem_ids={"leetcode-matched", "matched-source"},
    )

    assert [response.id for response in responses] == [
        "leetcode-no-source-id",
        "leetcode-snake-case-source",
    ]
    assert responses[0].sourceId is None
    assert responses[1].sourceId == "snake-source"


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
    assert concepts["Visited Array"]["kind"] == "data_structure"
    assert concepts["Queue"]["description"]

    assert any(problem["source"] == "UVa" for problem in payload["similarProblems"])
    assert any(problem["source"] == "LeetCode" for problem in payload["similarProblems"])
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
        assert {"BFS", "Queue", "Visited Array"}.issubset(problem["sharedConcepts"])
        assert problem["answerHint"]

    assert "\u7121\u6b0a\u5716\u4e2d\u627e\u6700\u77ed\u6b65\u6578" in payload["similarityReason"]
    assert "BFS \u627e\u6700\u77ed\u6b65\u6578" in payload["similarityReason"]
    assert any("\u5148\u5efa\u5716" in hint and "BFS" in hint for hint in payload["solvingHints"])
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
        title="Reverse Prefix of Word",
        source="LeetCode",
        score=0.72,
        text="Reverse the prefix of a word up to a target character.",
        concepts=(),
        problem_type="String",
        payload={
            "documentSource": "LeetCode",
            "sourceId": "2000",
            "answer": "Find the index and reverse that prefix.",
            "solutionHints": ("Use slicing.",),
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
        json={"input": "UVA-10653 - Bombs! NO they are Mines!!"},
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
    top_level_similar_problem_ids = [
        problem["id"] for problem in payload["similarProblems"]
    ]
    assert top_level_similar_problem_ids == ["leetcode-1091", "leetcode-994"]
    assert [problem["sourceId"] for problem in payload["similarProblems"]] == ["1091", "994"]
    assert {"uva-10653", "10653"}.isdisjoint(top_level_similar_problem_ids)


def test_analysis_uses_canonical_problem_ids_across_response_surfaces():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={
            "input": "UVA-10653 - Bombs! NO they are Mines!!",
            "mode": "hybrid",
            "topK": 3,
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
    assert {problem["sourceId"] for problem in top_level_similar_problems} == {"1091", "994"}
    assert {"uva-10653", "10653"}.isdisjoint(top_level_ids)

    evidence_ids = canonical_ids(
        payload["evidenceBundle"]["similarProblems"],
        id_key="id",
    )
    reranker_ids = canonical_ids(
        payload["retrievalTrace"]["rerankerScores"],
        id_key="id",
    )
    assert {"10653", "1091", "994"}.isdisjoint(evidence_ids | reranker_ids)
    assert top_level_ids == evidence_ids
    assert top_level_ids < reranker_ids
    assert payload["matchedProblem"]["id"] == "uva-10653"
    assert "uva-10653" not in evidence_ids
    assert "uva-10653" in reranker_ids


def test_analysis_exact_source_id_vector_query_preserves_retrieval_similar_problems():
    client = TestClient(app)

    response = client.post(
        "/api/analysis",
        json={"input": "10653", "mode": "vector"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matchedProblem"]["id"] == "uva-10653"
    assert payload["matchedProblem"]["sourceId"] == "10653"
    top_level_similar_problems = payload["similarProblems"]
    assert {
        (problem["source"], problem["id"], problem["title"])
        for problem in top_level_similar_problems
    } == {
        ("LeetCode", "leetcode-1091", "Shortest Path in Binary Matrix"),
        ("LeetCode", "leetcode-994", "Rotting Oranges"),
    }
    assert {problem["sourceId"] for problem in top_level_similar_problems} == {"1091", "994"}
    assert {"uva-10653", "10653"}.isdisjoint(
        {problem["id"] for problem in top_level_similar_problems}
    )
    assert all("Retrieved by selected mode" not in problem["reason"] for problem in top_level_similar_problems)
    assert all(
        problem["reason"].startswith("所選檢索模式將此題列為最終候選")
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
        == "No programming problem, code, concept, or retrieval evidence was detected."
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
    assert payload["matchedProblem"] is None
    assert "BFS" not in response.text


def test_analysis_recognized_unsupported_concept_abstains_consistently():
    client = TestClient(app)

    response = client.post("/api/analysis?debug=true", json={"input": "dynamic programming"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unsupported"
    assert payload["inputKind"] == "unknown"
    assert payload["problemType"] == ""
    assert payload["similarityReason"] == ""
    assert payload["abstentionReason"] == "This input is outside the supported graph traversal analysis scope."
    assert payload["requiredConcepts"] == []
    assert payload["similarProblems"] == []
    assert payload["matchedProblem"] is None
    assert payload["retrievalTrace"]["entityLinking"] == []
    assert payload["retrievalTrace"]["vectorCandidates"] == []
    assert payload["retrievalTrace"]["fusionScores"] == []
    assert payload["evidenceBundle"]["similarProblems"] == []


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

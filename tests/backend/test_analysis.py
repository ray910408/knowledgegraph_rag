from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app.analysis import find_graph_traversal_examples, load_programming_dataset
from backend.app.contracts import RetrievalTrace
from backend.app.main import AnalysisRequest, _analysis_paths_from_graph_trace, app
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
        assert set(problem) == {"source", "id", "title", "reason", "sharedConcepts", "answerHint"}
        assert {"BFS", "Queue", "Visited Array"}.issubset(problem["sharedConcepts"])
        assert problem["answerHint"]

    assert "\u7121\u6b0a\u5716\u4e2d\u627e\u6700\u77ed\u6b65\u6578" in payload["similarityReason"]
    assert "BFS \u627e\u6700\u77ed\u6b65\u6578" in payload["similarityReason"]
    assert any("\u5148\u5efa\u5716" in hint and "BFS" in hint for hint in payload["solvingHints"])
    assert any("visited" in mistake and "\u5fd8\u8a18\u6a19\u8a18" in mistake for mistake in payload["commonMistakes"])
    assert any("queue \u521d\u59cb\u5316\u932f\u8aa4" in mistake for mistake in payload["commonMistakes"])

    evidence = payload["evidencePaths"][0]
    assert set(evidence) == {"title", "nodes", "edges"}
    assert evidence["nodes"]
    assert evidence["edges"]
    assert {"id", "label", "type"}.issubset(evidence["nodes"][0])
    assert {"from", "to", "relation", "weight"}.issubset(evidence["edges"][0])
    relations = {edge["relation"] for edge in evidence["edges"]}
    assert "\u7b26\u5408\u8f38\u5165\u8a0a\u865f" in relations
    assert "\u9700\u8981\u89c0\u5ff5" in relations


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
    assert [problem["id"] for problem in payload["similarProblems"]] == ["1091", "2000"]
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
    assert top_level_similar_problem_ids == ["1091", "994"]
    assert {"uva-10653", "10653"}.isdisjoint(top_level_similar_problem_ids)


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
        ("LeetCode", "1091", "Shortest Path in Binary Matrix"),
        ("LeetCode", "994", "Rotting Oranges"),
    }
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

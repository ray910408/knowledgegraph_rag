from fastapi.testclient import TestClient

from backend.app.analysis import find_graph_traversal_examples, load_programming_dataset
from backend.app.main import app


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
    }


def test_analysis_response_includes_trace_and_evidence_without_context_by_default():
    client = TestClient(app)

    response = client.post("/api/analysis", json={"input": "unweighted graph shortest path BFS"})

    assert response.status_code == 200
    payload = response.json()
    assert "retrievalTrace" in payload
    assert "evidenceBundle" in payload
    assert "contextPreview" not in payload
    assert payload["retrievalTrace"]["queryUnderstanding"]["intent"] == "problem_search"
    assert payload["evidenceBundle"]["similarProblems"]


def test_analysis_debug_mode_includes_context_preview():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "unweighted graph shortest path BFS"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "contextPreview" in payload
    assert "Query Understanding" in payload["contextPreview"]


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

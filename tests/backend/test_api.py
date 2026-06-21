from fastapi.testclient import TestClient

from backend.app.main import app


def test_recommendations_endpoint_returns_techniques_without_dataset():
    client = TestClient(app)

    response = client.post(
        "/api/recommendations",
        json={
            "problemText": (
                "Given non-negative integers, find the shortest contiguous "
                "subarray whose sum is at least target."
            ),
            "mode": "hybrid",
            "topK": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["usedMockData"] is False
    assert payload["recommendations"]
    assert {item["kind"] for item in payload["recommendations"]} <= {
        "algorithm",
        "data_structure",
        "pattern",
    }
    assert payload["evidencePaths"]


def test_recommendations_endpoint_accepts_snake_case_and_v1_route():
    client = TestClient(app)

    response = client.post(
        "/api/v1/recommendations",
        json={
            "statement": "Count subarrays by target sum using prefix differences.",
            "mode": "graph",
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["queryId"] == "demo-graph-2"
    assert len(payload["recommendations"]) <= 2


def test_recommendations_endpoint_accepts_problem_id():
    client = TestClient(app)

    response = client.post(
        "/api/recommendations",
        json={"problemId": "demo-shortest-subarray", "mode": "hybrid", "topK": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["queryId"] == "demo-shortest-subarray-hybrid-2"
    assert payload["recommendations"]


def test_problem_id_with_text_marks_query_id_as_text_override():
    client = TestClient(app)

    response = client.post(
        "/api/recommendations",
        json={
            "problemId": "demo-shortest-subarray",
            "problemText": "Count subarrays by target sum using prefix differences.",
            "mode": "hybrid",
            "topK": 2,
        },
    )

    assert response.status_code == 200
    assert response.json()["queryId"] == "demo-shortest-subarray-with-text-hybrid-2"


def test_recommendations_endpoint_rejects_empty_statement():
    client = TestClient(app)

    response = client.post(
        "/api/recommendations",
        json={"problemText": "", "mode": "hybrid", "topK": 3},
    )

    assert response.status_code == 400


def test_health_has_versioned_route():
    client = TestClient(app)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

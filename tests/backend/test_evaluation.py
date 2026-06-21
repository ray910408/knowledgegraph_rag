import pytest

from backend.app.evaluation import EvaluationCase, run_fixture_evaluation


def test_fixture_evaluation_compares_all_modes_without_external_services():
    results = run_fixture_evaluation(
        (
            EvaluationCase(
                id="case-window",
                statement="Find shortest contiguous subarray with non-negative numbers.",
                expected_ids=("sliding-window",),
            ),
            EvaluationCase(
                id="case-prefix",
                statement="Count subarrays by target sum using prefix differences.",
                expected_ids=("prefix-sum", "hash-map"),
            ),
        ),
        top_k=3,
    )

    assert [item.mode for item in results] == ["vector", "graph", "hybrid"]
    assert all(item.cases == 2 for item in results)
    assert all(0 <= item.hit_rate <= 1 for item in results)
    assert results[2].hit_rate > 0


def test_fixture_evaluation_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        run_fixture_evaluation((), top_k=0)

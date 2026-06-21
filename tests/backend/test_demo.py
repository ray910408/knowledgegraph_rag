import pytest

from backend.app.demo import build_demo_repositories, recommend_demo_techniques


def test_demo_technique_inference_uses_token_boundaries():
    graph, _ = build_demo_repositories()

    recommendations = recommend_demo_techniques(
        graph,
        problem_text="Analyze paragraphs and formatting.",
        top_k=3,
        mode="hybrid",
    )

    assert all("graph" not in signal.lower() for item in recommendations for signal in item.fit_signals)


def test_demo_technique_recommendation_rejects_invalid_top_k():
    graph, _ = build_demo_repositories()

    with pytest.raises(ValueError, match="top_k"):
        recommend_demo_techniques(
            graph,
            problem_text="Find a range sum.",
            top_k=0,
            mode="hybrid",
        )

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.retrieval.documents import (
    ProcessedProblemDocumentLoader,
    ProcessedProblemDocumentLoaderError,
)


def _base_processed_problem() -> dict[str, object]:
    return {
        "id": "leetcode-994",
        "source": "LeetCode",
        "sourceId": "994",
        "title": "Processed Rotting Oranges",
        "problemType": "Graph Traversal",
        "statement": "Multi-source BFS with a queue on a grid.",
        "answer": "Use BFS from all rotten oranges.",
        "solutionHints": ["Push all rotten oranges first.", "Track minutes by BFS level."],
        "concepts": ["BFS", "Queue", "Visited Array"],
        "metadata": {"difficulty": "medium", "graphKind": "unweighted grid"},
        "constraints": ["1 <= m, n <= 10"],
        "examples": [{"input": "[[2,1,1],[1,1,0],[0,1,1]]", "output": "4"}],
        "editorial": "Every minute is one BFS layer from all starting rotten cells.",
    }


def _write_processed_problem(path: Path) -> None:
    path.write_text(
        json.dumps(
            [_base_processed_problem()],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_processed_problem_loader_converts_processed_json_to_retrieval_documents(tmp_path):
    processed_path = tmp_path / "problems.json"
    _write_processed_problem(processed_path)

    documents = ProcessedProblemDocumentLoader(processed_path).load()

    assert len(documents) == 1
    document = documents[0]
    assert document.id == "leetcode-994"
    assert document.source == "LeetCode"
    assert document.source_id == "994"
    assert document.title == "Processed Rotting Oranges"
    assert document.text == "Multi-source BFS with a queue on a grid."
    assert document.answer == "Use BFS from all rotten oranges."
    assert document.solution_hints == ("Push all rotten oranges first.", "Track minutes by BFS level.")
    assert document.concepts == ("BFS", "Queue", "Visited Array")
    assert document.problem_type == "Graph Traversal"
    assert document.difficulty == "medium"
    assert document.constraints == ("1 <= m, n <= 10",)
    assert document.examples == ({"input": "[[2,1,1],[1,1,0],[0,1,1]]", "output": "4"},)
    assert document.editorial == "Every minute is one BFS layer from all starting rotten cells."
    assert document.metadata["graphKind"] == "unweighted grid"


def test_processed_problem_loader_accepts_wrapped_problems_payload(tmp_path):
    processed_path = tmp_path / "problems.json"
    raw_problem = {
        "id": "uva-10653",
        "source": "UVa",
        "sourceId": "10653",
        "title": "Bombs",
        "problemType": "Graph Traversal",
        "statement": "Avoid blocked cells with BFS.",
        "answer": "Run BFS over free cells.",
        "solutionHints": [],
        "concepts": ["BFS"],
        "metadata": {},
    }
    processed_path.write_text(
        json.dumps({"problems": [raw_problem]}, ensure_ascii=False),
        encoding="utf-8",
    )

    documents = ProcessedProblemDocumentLoader(processed_path).load()

    assert documents[0].id == "uva-10653"
    assert documents[0].solution_hints == ()
    assert documents[0].constraints == ()
    assert documents[0].examples == ()


def test_processed_problem_loader_rejects_missing_file(tmp_path):
    processed_path = tmp_path / "missing.json"

    with pytest.raises(ProcessedProblemDocumentLoaderError, match="processed problems not found"):
        ProcessedProblemDocumentLoader(processed_path).load()


def test_processed_problem_loader_rejects_invalid_payload(tmp_path):
    processed_path = tmp_path / "problems.json"
    processed_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ProcessedProblemDocumentLoaderError, match="must be a list"):
        ProcessedProblemDocumentLoader(processed_path).load()


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("concepts", {"bad": "shape"}),
        ("solutionHints", [{"bad": "shape"}]),
        ("examples", ["not a mapping"]),
        ("metadata", ["bad"]),
        ("title", {"bad": "shape"}),
    ],
)
def test_processed_problem_loader_rejects_malformed_problem_fields(
    tmp_path, field_name, bad_value
):
    processed_path = tmp_path / "problems.json"
    raw_problem = _base_processed_problem()
    raw_problem[field_name] = bad_value
    processed_path.write_text(
        json.dumps([raw_problem], ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(
        ProcessedProblemDocumentLoaderError,
        match=rf"index 0.*{field_name}",
    ):
        ProcessedProblemDocumentLoader(processed_path).load()

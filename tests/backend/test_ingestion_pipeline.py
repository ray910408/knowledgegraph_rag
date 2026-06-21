from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from backend.app.ingestion.pipeline import build_ingestion_artifacts


def _write_raw_problem(path: Path) -> None:
    payload = {
        "problems": [
            {
                "id": "leetcode-994-a",
                "source": "LeetCode",
                "sourceId": "994",
                "title": " Rotting   Oranges ",
                "problemType": "Graph Traversal",
                "statement": "Multi-source   BFS\nwith a queue.",
                "answer": "Use BFS from all rotten oranges.",
                "solutionHints": ["Push all sources first."],
                "concepts": ["BFS", "Queue"],
                "tags": ["matrix"],
                "metadata": {"difficulty": "medium"},
            },
            {
                "id": "leetcode-994-b",
                "source": "LeetCode",
                "sourceId": "994",
                "title": "Duplicate should be ignored",
                "problemType": "Graph Traversal",
                "statement": "duplicate",
                "answer": "duplicate",
                "solutionHints": [],
                "concepts": ["BFS"],
                "tags": [],
                "metadata": {},
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ingestion_builds_processed_json_and_search_artifacts(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_raw_problem(raw_dir / "problems.json")

    manifest = build_ingestion_artifacts(
        input_dir=raw_dir,
        processed_dir=processed_dir,
        target="all",
        allow_fallback=True,
    )

    assert manifest["target"] == "all"
    assert manifest["counts"]["problems"] == 1
    assert manifest["counts"]["chunks"] >= 2
    assert manifest["counts"]["entities"] >= 3
    assert manifest["fallback"]["qdrant"] is True
    assert manifest["fallback"]["neo4j"] is True

    problems = json.loads((processed_dir / "problems.json").read_text(encoding="utf-8"))
    chunks = json.loads((processed_dir / "chunks.json").read_text(encoding="utf-8"))
    entities = json.loads((processed_dir / "entities.json").read_text(encoding="utf-8"))
    relations = json.loads((processed_dir / "relations.json").read_text(encoding="utf-8"))
    bm25 = json.loads((processed_dir / "bm25_index.json").read_text(encoding="utf-8"))
    qdrant = json.loads((processed_dir / "qdrant_vectors.json").read_text(encoding="utf-8"))
    neo4j = json.loads((processed_dir / "neo4j_graph.json").read_text(encoding="utf-8"))

    assert problems[0]["id"] == "leetcode-994-a"
    assert problems[0]["title"] == "Rotting Oranges"
    assert problems[0]["statement"] == "Multi-source BFS with a queue."
    assert {chunk["kind"] for chunk in chunks} >= {"statement", "answer"}
    assert "concept:bfs" in {entity["id"] for entity in entities}
    assert any(relation["type"] == "REQUIRES" for relation in relations)
    assert bm25["documents"][0]["id"] == chunks[0]["id"]
    assert qdrant["embeddingModel"] == "BAAI/bge-m3"
    assert qdrant["records"][0]["vector"]
    assert neo4j["entities"] == entities
    assert neo4j["relations"] == relations


def test_ingestion_cli_supports_json_target(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_raw_problem(raw_dir / "problems.json")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.ingestion",
            "build",
            "--input",
            str(raw_dir),
            "--processed",
            str(processed_dir),
            "--target",
            "json",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (processed_dir / "problems.json").exists()
    assert (processed_dir / "chunks.json").exists()
    assert not (processed_dir / "qdrant_vectors.json").exists()


def test_ingestion_cli_requires_fallback_for_docker_targets(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_raw_problem(raw_dir / "problems.json")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.ingestion",
            "build",
            "--input",
            str(raw_dir),
            "--processed",
            str(processed_dir),
            "--target",
            "qdrant",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "Qdrant is not available" in completed.stderr
    assert "--allow-fallback" in completed.stderr

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from backend.app.ingestion.pipeline import build_ingestion_artifacts


def test_committed_programming_problem_seed_is_readable_utf8():
    seed_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "programming_problems.json"
    seed_text = seed_path.read_text(encoding="utf-8")

    assert not seed_text.startswith("\ufeff")
    assert "\ufffd" not in seed_text
    assert "\\u" not in seed_text
    assert "在有障礙的無權網格中" in seed_text
    assert len(json.loads(seed_text)["problems"]) == 3


def _write_raw_problem(path: Path) -> None:
    payload = {
        "problems": [
            {
                "id": "leetcode-994-a",
                "source": "LeetCode",
                "sourceId": "994",
                "title": " Rotting   Oranges ",
                "problemType": "Graph Traversal",
                "statement": "多源 BFS\n使用 Queue 逐層擴散。",
                "answer": "從所有腐爛橘子同時開始 BFS。",
                "solutionHints": ["先把所有起點放入 Queue。"],
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    assert "manifest.json" in manifest["artifacts"]

    problems_text = (processed_dir / "problems.json").read_text(encoding="utf-8")
    assert "\\u" not in problems_text
    assert "從所有腐爛橘子同時開始 BFS。" in problems_text

    problems = json.loads(problems_text)
    chunks = json.loads((processed_dir / "chunks.json").read_text(encoding="utf-8"))
    entities = json.loads((processed_dir / "entities.json").read_text(encoding="utf-8"))
    relations = json.loads((processed_dir / "relations.json").read_text(encoding="utf-8"))
    bm25 = json.loads((processed_dir / "bm25_index.json").read_text(encoding="utf-8"))
    qdrant = json.loads((processed_dir / "qdrant_vectors.json").read_text(encoding="utf-8"))
    neo4j = json.loads((processed_dir / "neo4j_graph.json").read_text(encoding="utf-8"))
    assert (processed_dir / "manifest.json").exists()

    assert problems[0]["id"] == "leetcode-994-a"
    assert problems[0]["title"] == "Rotting Oranges"
    assert problems[0]["statement"] == "多源 BFS 使用 Queue 逐層擴散。"
    assert {chunk["kind"] for chunk in chunks} >= {"statement", "answer"}
    assert "concept:bfs" in {entity["id"] for entity in entities}
    assert any(relation["type"] == "REQUIRES" for relation in relations)
    assert bm25["documents"][0]["id"] == chunks[0]["id"]
    assert qdrant["embeddingModel"] == "BAAI/bge-m3"
    assert qdrant["records"][0]["vector"]
    assert neo4j["entities"] == entities
    assert neo4j["relations"] == relations


def test_ingestion_enriches_vector_and_bm25_payloads(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    raw_payload = {
        "problems": [
            {
                "id": "leetcode-994",
                "source": "LeetCode",
                "sourceId": "994",
                "title": "Rotting Oranges",
                "problemType": "Graph Traversal",
                "statement": "Multi-source BFS with a queue on a grid.",
                "answer": "Use BFS from all rotten oranges.",
                "solutionHints": ["Push all rotten oranges first."],
                "concepts": ["BFS", "Queue"],
                "tags": ["matrix"],
                "metadata": {"graphKind": "unweighted grid"},
                "difficulty": "Medium",
                "constraints": ["1 <= m, n <= 10"],
                "examples": [{"input": "grid", "output": "4"}],
                "editorial": "Each BFS layer is one minute.",
            }
        ]
    }
    (raw_dir / "problems.json").write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    build_ingestion_artifacts(
        input_dir=raw_dir,
        processed_dir=processed_dir,
        target="all",
        allow_fallback=True,
    )

    bm25 = json.loads((processed_dir / "bm25_index.json").read_text(encoding="utf-8"))
    qdrant = json.loads((processed_dir / "qdrant_vectors.json").read_text(encoding="utf-8"))
    for payload in (
        bm25["documents"][0]["payload"],
        qdrant["records"][0]["payload"],
    ):
        assert payload["answer"] == "Use BFS from all rotten oranges."
        assert payload["solutionHints"] == ["Push all rotten oranges first."]
        assert payload["difficulty"] == "Medium"
        assert payload["constraints"] == ["1 <= m, n <= 10"]
        assert payload["examples"] == [{"input": "grid", "output": "4"}]
        assert payload["editorial"] == "Each BFS layer is one minute."
        assert payload["source"] == "LeetCode"
        assert payload["sourceId"] == "994"
        assert payload["title"] == "Rotting Oranges"
        assert payload["problemType"] == "Graph Traversal"
        assert payload["concepts"] == ["BFS", "Queue"]


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
    fake_modules_dir = tmp_path / "fake-modules"
    fake_qdrant_client = fake_modules_dir / "qdrant_client"
    raw_dir.mkdir()
    fake_qdrant_client.mkdir(parents=True)
    _write_raw_problem(raw_dir / "problems.json")
    (fake_qdrant_client / "__init__.py").write_text(
        'raise ImportError("qdrant-client intentionally unavailable for this test")\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_modules_dir) + os.pathsep + env.get("PYTHONPATH", "")

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
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "Qdrant is not available" in completed.stderr
    assert "--allow-fallback" in completed.stderr


def test_ingestion_can_write_to_injected_db_adapters_without_fallback(tmp_path):
    class CapturingVectorStore:
        def __init__(self):
            self.records = ()

        def upsert(self, records):
            self.records = tuple(records)

        def search(self, query_vector, *, top_k, filters=None):
            return ()

    class CapturingGraphStore:
        def __init__(self):
            self.entities = ()
            self.relations = ()

        def upsert_entities(self, entities):
            self.entities = tuple(entities)

        def upsert_relations(self, relations):
            self.relations = tuple(relations)

        def find_paths(self, source_id, target_id, *, max_hops=3):
            return ()

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_raw_problem(raw_dir / "problems.json")
    vector_store = CapturingVectorStore()
    graph_store = CapturingGraphStore()

    manifest = build_ingestion_artifacts(
        input_dir=raw_dir,
        processed_dir=processed_dir,
        target="all",
        allow_fallback=False,
        vector_store=vector_store,
        graph_store=graph_store,
    )

    assert manifest["fallback"] == {"qdrant": False, "neo4j": False}
    assert vector_store.records
    assert graph_store.entities
    assert graph_store.relations

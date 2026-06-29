from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from backend.app.chunking.search_text import build_chunk_search_text
from backend.app.contracts import RawProblem
from backend.app.chunking.router import ChunkingRouter
from backend.app.ingestion.pipeline import _build_chunks, build_ingestion_artifacts
from backend.app.providers import DeterministicMockEmbeddingProvider


def test_committed_programming_problem_seed_is_readable_utf8():
    seed_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "programming_problems.json"
    seed_text = seed_path.read_text(encoding="utf-8")

    assert not seed_text.startswith("\ufeff")
    assert "\ufffd" not in seed_text
    assert "\\u" not in seed_text
    seed_problem_ids = {problem["id"] for problem in json.loads(seed_text)["problems"]}
    assert {"uva-10653", "leetcode-1091", "leetcode-994"} <= seed_problem_ids


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


def _structured_problem(**overrides) -> RawProblem:
    payload = {
        "id": "leetcode-994",
        "source": "LeetCode",
        "source_id": "994",
        "title": "Rotting Oranges",
        "problem_type": "Graph Traversal",
        "statement": "Use BFS to spread rot across the grid.",
        "answer": "Run multi-source BFS from the rotten oranges.",
        "solution_hints": ("Push all rotten oranges first.", "Expand one BFS layer per minute."),
        "concepts": ("BFS", "Queue"),
        "tags": ("matrix", "graph"),
        "metadata": {"difficulty": "Medium"},
        "constraints": ("1 <= m, n <= 10",),
        "examples": ({"input": "grid = [[2,1,1],[1,1,0],[0,1,1]]", "output": "4"},),
        "editorial": "Each BFS layer represents one minute.",
    }
    payload.update(overrides)
    return RawProblem(**payload)


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
    assert {chunk["kind"] for chunk in chunks} >= {"problem_card", "statement", "solution"}
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
    neo4j = json.loads((processed_dir / "neo4j_graph.json").read_text(encoding="utf-8"))
    statement_bm25 = next(
        document for document in bm25["documents"] if document["payload"]["kind"] == "statement"
    )
    statement_qdrant = next(
        record for record in qdrant["records"] if record["payload"]["kind"] == "statement"
    )
    search_text = statement_bm25["text"]
    for payload in (
        statement_bm25["payload"],
        statement_qdrant["payload"],
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

    assert "Graph Traversal" in search_text
    assert "graph traversal" in search_text
    assert "圖論遍歷" in search_text
    assert "圖遍歷" in search_text
    assert "廣度優先搜尋" in search_text
    assert "廣搜" in search_text
    assert "佇列" in search_text
    assert "隊列" in search_text
    assert statement_bm25["text"] == statement_bm25["payload"]["searchText"]
    assert statement_qdrant["payload"]["text"] == "Multi-source BFS with a queue on a grid."
    assert statement_qdrant["payload"]["displayText"] == "Multi-source BFS with a queue on a grid."
    assert statement_qdrant["payload"]["searchText"] == search_text
    assert statement_qdrant["vector"] == list(
        DeterministicMockEmbeddingProvider().embed_text(search_text)
    )

    bfs_entity = next(entity for entity in neo4j["entities"] if entity["id"] == "concept:bfs")
    assert "廣搜" in bfs_entity["aliases"]
    assert "廣度優先搜尋" in bfs_entity["aliases"]


def test_build_chunks_skips_whitespace_only_fields_before_serialization_output():
    problem = RawProblem(
        id="leetcode-994",
        source="LeetCode",
        source_id="994",
        title="Rotting Oranges",
        problem_type="Graph Traversal",
        statement="   \n\t  ",
        answer="Use BFS from all rotten oranges.",
        solution_hints=("   ", "\t"),
        concepts=("BFS",),
        editorial="  \n  ",
    )

    serialized_chunks = [chunk.to_mapping() for chunk in _build_chunks((problem,))]

    assert [chunk["kind"] for chunk in serialized_chunks] == ["problem_card", "solution"]
    assert serialized_chunks[1]["text"] == "Use BFS from all rotten oranges."


def test_build_chunks_routes_raw_problem_through_chunking_router(monkeypatch):
    calls: list[tuple[RawProblem, str]] = []

    class SpyRouter:
        def chunk_problem(self, problem, *, runtime_type="structured_problem"):
            calls.append((problem, runtime_type))
            return (
                RawProblem,  # sentinel to prove _build_chunks uses the router output directly
            )

    monkeypatch.setattr(
        "backend.app.ingestion.pipeline.ChunkingRouter",
        SpyRouter,
        raising=False,
    )

    chunks = _build_chunks((_structured_problem(),))

    assert calls == [(_structured_problem(), "structured_problem")]
    assert chunks == (RawProblem,)


def test_structured_problem_chunking_emits_problem_card_and_merged_hints():
    serialized_chunks = [chunk.to_mapping() for chunk in _build_chunks((_structured_problem(),))]
    kinds = [chunk["kind"] for chunk in serialized_chunks]

    assert kinds == [
        "problem_card",
        "statement",
        "constraints",
        "examples",
        "hints",
        "solution",
    ]
    assert serialized_chunks[0]["id"] == "leetcode-994:problem_card:0"
    assert "LeetCode 994" in serialized_chunks[0]["text"]
    assert "Rotting Oranges" in serialized_chunks[0]["text"]
    assert "Graph Traversal" in serialized_chunks[0]["text"]
    assert "Medium" in serialized_chunks[0]["text"]
    assert "matrix" in serialized_chunks[0]["text"]
    assert serialized_chunks[4]["id"] == "leetcode-994:hints:0"
    assert serialized_chunks[4]["text"] == (
        "Push all rotten oranges first.\nExpand one BFS layer per minute."
    )


def test_build_chunk_search_text_uses_display_lane_only():
    problem = _structured_problem(
        metadata={
            "difficulty": "Medium",
            "commonMistakes": ["Do not leak this template mistake into retrieval text."],
            "commonMistakesSource": "template",
        }
    )

    search_text = build_chunk_search_text(
        problem_id=problem.id,
        source=problem.source,
        source_id=problem.source_id,
        title=problem.title,
        problem_type=problem.problem_type,
        concepts=problem.concepts,
        display_text=problem.statement,
    )

    assert problem.statement in search_text
    assert "Rotting Oranges" in search_text
    assert "Graph Traversal" in search_text
    assert "graph traversal" in search_text
    assert "Do not leak this template mistake into retrieval text." not in search_text


def test_structured_problem_chunking_sets_display_and_search_text_lanes():
    chunks = [chunk.to_mapping() for chunk in _build_chunks((_structured_problem(),))]
    statement_chunk = next(chunk for chunk in chunks if chunk["kind"] == "statement")

    assert statement_chunk["text"] == "Use BFS to spread rot across the grid."
    assert statement_chunk["displayText"] == "Use BFS to spread rot across the grid."
    assert statement_chunk["searchText"] == build_chunk_search_text(
        problem_id="leetcode-994",
        source="LeetCode",
        source_id="994",
        title="Rotting Oranges",
        problem_type="Graph Traversal",
        concepts=("BFS", "Queue"),
        display_text="Use BFS to spread rot across the grid.",
    )


def test_structured_problem_chunking_skips_empty_optional_sections_and_template_common_mistakes():
    serialized_chunks = [
        chunk.to_mapping()
        for chunk in _build_chunks(
            (
                _structured_problem(
                    constraints=("   ", "\n\t"),
                    examples=(),
                    solution_hints=("   ", "\n"),
                    answer="  ",
                    editorial=" \n ",
                    metadata={
                        "difficulty": "Medium",
                        "commonMistakes": ["Do not mutate the template bullet list."],
                        "commonMistakesSource": "template",
                    },
                ),
            )
        )
    ]

    assert [chunk["kind"] for chunk in serialized_chunks] == ["problem_card", "statement"]
    assert all(
        "Do not mutate the template bullet list." not in chunk["searchText"]
        for chunk in serialized_chunks
    )


def test_chunking_router_module_supports_structured_problem_runtime_only():
    spec = importlib.util.find_spec("backend.app.chunking.router")

    assert spec is not None
    router_module = importlib.import_module("backend.app.chunking.router")
    assert router_module.ChunkingRouter.supported_runtime_types() == ("structured_problem",)


def test_chunking_router_dispatches_structured_problem_runtime():
    router = ChunkingRouter()

    chunks = router.chunk_problem(_structured_problem(), runtime_type="structured_problem")

    assert [chunk.kind for chunk in chunks] == [
        "problem_card",
        "statement",
        "constraints",
        "examples",
        "hints",
        "solution",
    ]


def test_chunking_router_rejects_unsupported_runtime_type():
    router = ChunkingRouter()

    try:
        router.chunk_problem(_structured_problem(), runtime_type="raw_problem")
    except ValueError as exc:
        assert str(exc) == "unsupported chunking runtime type: raw_problem"
    else:
        raise AssertionError("expected ValueError for unsupported runtime type")


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

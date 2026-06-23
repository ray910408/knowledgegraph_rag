from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.app.retrieval.runtime import RuntimeRetrievalSettings, build_runtime_retrieval


pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(
        os.getenv("RUN_DOCKER_TESTS") != "1",
        reason="set RUN_DOCKER_TESTS=1 after starting Docker services and running ingestion",
    ),
]


def test_store_runtime_can_query_seeded_qdrant_neo4j_and_bm25():
    settings = RuntimeRetrievalSettings(
        backend="stores",
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "programming_chunks"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        bm25_index_path=Path(os.getenv("BM25_INDEX_PATH", "data/processed/bm25_index.json")),
        processed_problems_path=Path(
            os.getenv("PROCESSED_PROBLEMS_PATH", "data/processed/problems.json")
        ),
    )

    runtime = build_runtime_retrieval(settings=settings)
    result = runtime.pipeline.run("unweighted graph shortest path BFS queue", top_k=3)
    trace = result.trace.to_mapping()

    assert runtime.backend == "stores"
    assert runtime.candidate_sources == {
        "vector": "qdrant",
        "graph": "neo4j",
        "bm25": "bm25_index",
    }
    assert trace["vectorCandidates"]
    assert trace["bm25Candidates"]
    assert trace["graphCandidates"]
    assert any("storePath" in path for path in result.graph_paths)

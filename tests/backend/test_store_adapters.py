from __future__ import annotations

import json

from backend.app.adapters.in_memory import InMemoryBM25Store, InMemoryGraphStore, InMemoryVectorStore
from backend.app.adapters.neo4j import Neo4jGraphStore
from backend.app.adapters.qdrant import QdrantVectorStore
from backend.app.contracts import EntityRecord, RelationRecord
from backend.app.stores import BM25Document, SearchCandidate, VectorRecord


def test_in_memory_vector_store_searches_by_cosine_similarity():
    store = InMemoryVectorStore()
    store.upsert(
        (
            VectorRecord(id="a", vector=(1.0, 0.0), payload={"title": "BFS"}),
            VectorRecord(id="b", vector=(0.0, 1.0), payload={"title": "DP"}),
        )
    )

    results = store.search((1.0, 0.0), top_k=2)

    assert [result.id for result in results] == ["a", "b"]
    assert results[0].score == 1.0
    assert results[0].payload["title"] == "BFS"


def test_in_memory_bm25_store_returns_keyword_matches():
    store = InMemoryBM25Store()
    store.index_documents(
        (
            BM25Document(id="bfs", text="BFS shortest path queue"),
            BM25Document(id="dp", text="dynamic programming subsequence"),
        )
    )

    results = store.search("shortest path BFS", top_k=2)

    assert results[0].id == "bfs"
    assert results[0].score > results[1].score


def test_in_memory_graph_store_returns_paths():
    store = InMemoryGraphStore()
    store.upsert_entities(
        (
            EntityRecord(id="leetcode-994", name="Rotting Oranges", type="problem"),
            EntityRecord(id="concept:bfs", name="BFS", type="algorithm"),
        )
    )
    store.upsert_relations(
        (
            RelationRecord(
                id="leetcode-994->concept:bfs",
                source_id="leetcode-994",
                target_id="concept:bfs",
                type="REQUIRES",
            ),
        )
    )

    paths = store.find_paths("leetcode-994", "concept:bfs")

    assert paths == (
        {
            "nodes": ["leetcode-994", "concept:bfs"],
            "relations": ["REQUIRES"],
            "score": 1.0,
        },
    )


def test_qdrant_vector_store_uses_injected_client_without_docker():
    class Point:
        id = "chunk-1"
        score = 0.9
        payload = {"problemId": "leetcode-994"}

    class QueryResult:
        points = [Point()]

    class FakeClient:
        def __init__(self):
            self.upserts = []

        def upsert(self, collection_name, points):
            self.upserts.append((collection_name, points))

        def query_points(self, collection_name, query, limit, query_filter=None):
            return QueryResult()

    client = FakeClient()
    store = QdrantVectorStore(client=client, collection_name="problems")
    store.upsert((VectorRecord(id="chunk-1", vector=(0.1, 0.2), payload={}),))
    results = store.search((0.1, 0.2), top_k=1)

    assert client.upserts
    assert results == (SearchCandidate(id="chunk-1", score=0.9, payload={"problemId": "leetcode-994"}),)


def test_neo4j_graph_store_uses_injected_driver_without_docker():
    class FakeSession:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, query, **params):
            self.calls.append((query, params))
            if "MATCH path" in query:
                return [{"path": {"nodes": ["a", "b"], "relations": ["REQUIRES"], "score": 1.0}}]
            return []

    class FakeDriver:
        def __init__(self):
            self.session_instance = FakeSession()

        def session(self):
            return self.session_instance

    driver = FakeDriver()
    store = Neo4jGraphStore(driver=driver)
    store.upsert_entities(
        (
            EntityRecord(
                id="concept:bfs",
                name="BFS",
                type="algorithm",
                aliases=("Breadth First Search",),
                problem_ids=("leetcode-994",),
                metadata={"source": "測試", "nested": {"level": 1}},
            ),
        )
    )
    store.upsert_relations(
        (
            RelationRecord(
                id="leetcode-994->concept:bfs",
                source_id="leetcode-994",
                target_id="concept:bfs",
                type="REQUIRES",
                weight=0.8,
                evidence=("uses queue",),
                metadata={"origin": "測試", "nested": {"safe": True}},
            ),
        )
    )
    paths = store.find_paths("a", "b")

    assert paths == ({"nodes": ["a", "b"], "relations": ["REQUIRES"], "score": 1.0},)
    assert len(driver.session_instance.calls) == 3
    entity_query, entity_params = driver.session_instance.calls[0]
    relation_query, relation_params = driver.session_instance.calls[1]
    entity_row = entity_params["entities"][0]
    relation_row = relation_params["relations"][0]

    assert "SET n += entity" not in entity_query
    assert "metadata = relation.metadata" not in relation_query
    assert set(entity_row) == {"id", "name", "type", "aliases", "problemIds", "metadataJson"}
    assert set(relation_row) == {
        "id",
        "sourceId",
        "targetId",
        "type",
        "weight",
        "evidence",
        "metadataJson",
    }
    assert "metadata" not in entity_row
    assert "metadata" not in relation_row
    assert not any(isinstance(value, dict) for value in entity_row.values())
    assert not any(isinstance(value, dict) for value in relation_row.values())
    assert entity_row["metadataJson"] == json.dumps(
        {"source": "測試", "nested": {"level": 1}}, ensure_ascii=False, sort_keys=True
    )
    assert relation_row["metadataJson"] == json.dumps(
        {"origin": "測試", "nested": {"safe": True}}, ensure_ascii=False, sort_keys=True
    )

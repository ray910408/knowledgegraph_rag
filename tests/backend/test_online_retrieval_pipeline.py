from __future__ import annotations

from dataclasses import replace
import json

import pytest

from backend.app.adapters.in_memory import (
    InMemoryBM25Store,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from backend.app.contracts import EntityRecord, RelationRecord
from backend.app.providers import DeterministicMockEmbeddingProvider
from backend.app.retrieval import pipeline as retrieval_pipeline
from backend.app.retrieval.pipeline import (
    BM25SearchService,
    CodeFeatureExtractor,
    ContextBuilder,
    EntityLinkingService,
    EvidenceBuilder,
    ExactProblemMatch,
    ExactProblemMatcher,
    GraphSearchService,
    HybridFusionService,
    OnlineQueryPipeline,
    QueryUnderstandingService,
    Reranker,
    RetrievalCandidate,
    RetrievalDocument,
    VectorSearchService,
    _aggregate_problem_candidates,
    _graph_edge,
)
from backend.app.stores import BM25Document, SearchCandidate, VectorRecord


CPP_BFS_SNIPPET = """
#include <bits/stdc++.h>
using namespace std;

int bfs(vector<vector<int>>& grid) {
    queue<pair<int, int>> q;
    vector<vector<int>> visited(grid.size(), vector<int>(grid[0].size(), 0));
    vector<pair<int, int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        for (auto [dr, dc] : directions) {
            int nr = r + dr;
            int nc = c + dc;
            if (!visited[nr][nc]) {
                visited[nr][nc] = 1;
                q.push({nr, nc});
            }
        }
    }
    return 0;
}
""".strip()

PYTHON_BFS_SNIPPET = """
from collections import deque

def bfs(grid):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    queue = deque([(0, 0)])
    seen = set([(0, 0)])
    while queue:
        row, col = queue.popleft()
        for dr, dc in directions:
            nr = row + dr
            nc = col + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen)
""".strip()


def _documents() -> tuple[RetrievalDocument, ...]:
    return (
        RetrievalDocument(
            id="leetcode-994",
            source="LeetCode",
            source_id="994",
            title="Rotting Oranges",
            text="Multi-source BFS with a queue on a grid.",
            answer="Use BFS from all rotten oranges.",
            concepts=("BFS", "Queue"),
            problem_type="Graph Traversal",
            solution_hints=("Push all rotten oranges first.", "Expand one BFS layer per minute."),
            difficulty="Medium",
            constraints=("1 <= m, n <= 10",),
            examples=({"input": "grid", "output": "4"},),
            editorial="Use multi-source BFS from all rotten cells.",
        ),
        RetrievalDocument(
            id="leetcode-300",
            source="LeetCode",
            source_id="300",
            title="Longest Increasing Subsequence",
            text="Dynamic programming over increasing subsequences.",
            answer="Use DP.",
            concepts=("Dynamic Programming",),
            problem_type="Dynamic Programming",
        ),
    )


def _queue_documents() -> tuple[RetrievalDocument, ...]:
    return (
        _documents()[0],
        RetrievalDocument(
            id="leetcode-1091",
            source="LeetCode",
            source_id="1091",
            title="Shortest Path in Binary Matrix",
            text="Use BFS with a queue to find the shortest path in a binary matrix.",
            answer="Run BFS over eight directions.",
            concepts=("BFS", "Queue"),
            problem_type="Graph Traversal",
        ),
    )


def _dynamic_programming_document() -> RetrievalDocument:
    return RetrievalDocument(
        id="uva-437",
        source="UVa",
        source_id="437",
        title="The Tower of Babylon",
        text="Rotate blocks and solve the maximum stack height with LIS-style DP.",
        answer="Treat each orientation as a block, sort by base area, then run DP.",
        concepts=("DP", "LIS", "Sorting"),
        problem_type="Dynamic Programming",
        solution_hints=(
            "把每個方塊展開成 3 種可用方向，讓每種方向都能成為底座。",
            "依底面尺寸排序後，用 dp[i] 表示以第 i 個方向作為頂端時的最大高度。",
        ),
        difficulty="Hard",
    )


def _store_payload(document: RetrievalDocument) -> dict[str, object]:
    return {
        "problemId": document.id,
        "kind": "statement",
        "text": document.text,
        "answer": document.answer,
        "solutionHints": list(document.solution_hints),
        "difficulty": document.difficulty,
        "constraints": list(document.constraints),
        "examples": [dict(example) for example in document.examples],
        "editorial": document.editorial,
        "source": document.source,
        "sourceId": document.source_id,
        "title": document.title,
        "problemType": document.problem_type,
        "concepts": list(document.concepts),
        "metadata": {
            "source": document.source,
            "sourceId": document.source_id,
            "title": document.title,
            "problemType": document.problem_type,
            "answer": document.answer,
        },
    }


def _build_vector_store(
    documents: tuple[RetrievalDocument, ...],
    embedding_provider: DeterministicMockEmbeddingProvider,
) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    records = []
    for document in documents:
        text = (
            "BFS shortest path with queue"
            if document.id == "leetcode-994"
            else f"{document.title} {document.text}"
        )
        records.append(
            VectorRecord(
                id=f"{document.id}:statement:0",
                vector=tuple(embedding_provider.embed_text(text)),
                payload=_store_payload(document),
            )
        )
    store.upsert(tuple(records))
    return store


def _build_bm25_store(documents: tuple[RetrievalDocument, ...]) -> InMemoryBM25Store:
    store = InMemoryBM25Store()
    store.index_documents(
        tuple(
            BM25Document(
                id=f"{document.id}:statement:0",
                text=f"{document.title} {document.text} {document.answer}",
                payload=_store_payload(document),
            )
            for document in documents
        )
    )
    return store


def _build_graph_store(documents: tuple[RetrievalDocument, ...]) -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.upsert_entities(
        (
            EntityRecord(id="concept:bfs", name="BFS", type="algorithm"),
            EntityRecord(id="concept:queue", name="Queue", type="data_structure"),
            EntityRecord(id="pattern:graph-traversal", name="Graph Traversal", type="pattern"),
            *(
                EntityRecord(
                    id=document.id,
                    name=document.title,
                    type="problem",
                    metadata={
                        "source": document.source,
                        "sourceId": document.source_id,
                        "problemType": document.problem_type,
                    },
                )
                for document in documents
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
                weight=1.0,
            ),
            RelationRecord(
                id="leetcode-994->concept:queue",
                source_id="leetcode-994",
                target_id="concept:queue",
                type="REQUIRES",
                weight=1.0,
            ),
            RelationRecord(
                id="leetcode-994->pattern:graph-traversal",
                source_id="leetcode-994",
                target_id="pattern:graph-traversal",
                type="HAS_PATTERN",
                weight=1.0,
            ),
        )
    )
    return store


def test_code_feature_extractor_maps_cpp_and_python_bfs_to_same_features():
    expected_features = ("bfs", "queue_frontier", "visited_state", "grid_traversal")

    cpp_features = CodeFeatureExtractor().extract(CPP_BFS_SNIPPET, input_kind="cpp")
    python_features = CodeFeatureExtractor().extract(PYTHON_BFS_SNIPPET, input_kind="python")

    assert cpp_features.language == "cpp"
    assert cpp_features.features == expected_features
    assert python_features.language == "python"
    assert python_features.features == expected_features


def test_code_queries_link_same_graph_concepts_for_cpp_and_python_bfs():
    expected_feature_nodes = {
        "code_feature:bfs",
        "code_feature:queue_frontier",
        "code_feature:visited_state",
    }

    for snippet, language in (
        (CPP_BFS_SNIPPET, "cpp"),
        (PYTHON_BFS_SNIPPET, "python"),
    ):
        understanding = QueryUnderstandingService().understand(snippet)
        linked_entities = EntityLinkingService().link(understanding)
        linked_by_name = {str(entity["name"]): entity for entity in linked_entities}
        code_feature_nodes = {
            str(entity["codeFeatureNodeId"])
            for entity in linked_entities
            if entity.get("matchedBy") == "code_feature"
        }

        assert understanding.input_kind == language
        assert understanding.code_features is not None
        assert understanding.code_features.language == language
        assert set(understanding.code_features.features) >= {
            "bfs",
            "queue_frontier",
            "visited_state",
        }
        assert {"BFS", "Queue", "Visited Array"} <= set(linked_by_name)
        assert expected_feature_nodes <= code_feature_nodes
        assert linked_by_name["BFS"]["matchedBy"] == "code_feature"
        assert linked_by_name["Queue"]["matchedBy"] == "code_feature"
        assert linked_by_name["Visited Array"]["matchedBy"] == "code_feature"


@pytest.mark.parametrize(
    "snippet",
    [
        "#include <queue>\nint main(){ std::queue<int> q; q.push(1); }",
        "def solve(nums):\n    seen=set(nums)\n    return len(seen)",
        "from collections import deque\ndef solve(a):\n    q=deque()\n    q.append(1)\n    return q.popleft()",
    ],
)
def test_code_feature_extractor_ignores_non_traversal_queue_deque_and_set(snippet: str):
    understanding = QueryUnderstandingService().understand(snippet)
    linked_entities = EntityLinkingService().link(understanding)
    linked_names = {str(entity["name"]) for entity in linked_entities}

    assert understanding.code_features is not None
    assert understanding.code_features.features == ()
    assert all(entity.get("matchedBy") != "code_feature" for entity in linked_entities)
    assert "BFS" not in linked_names
    assert "Visited Array" not in linked_names


def test_query_understanding_extracts_multilingual_terms_for_unweighted_shortest_path():
    query = "給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。"

    understanding = QueryUnderstandingService().understand(query)
    mapping = understanding.to_mapping()

    assert mapping["queryLanguage"] == "zh-Hant"
    assert {"無權圖", "最短步數"}.issubset(set(mapping["keywords"]))
    assert {"BFS", "Queue", "Shortest Path"}.issubset(set(mapping["conceptSeeds"]))
    assert "unweighted graph" in mapping["expandedTerms"]
    assert {"shortest path", "shortest steps"} & set(mapping["expandedTerms"])
    assert mapping["queryVariants"]["bm25"].startswith(query)
    assert "breadth first search" in mapping["queryVariants"]["bm25"]
    assert {
        "concept:bfs",
        "concept:queue",
        "concept:shortest-path",
    }.issubset(set(mapping["queryVariants"]["graphSeeds"]))


@pytest.mark.parametrize("query", ["DP", "dynamic programming", "動態規劃"])
def test_query_understanding_promotes_dynamic_programming_aliases_to_concept_seed(
    query: str,
):
    understanding = QueryUnderstandingService((_dynamic_programming_document(),)).understand(query)

    assert "Dynamic Programming" in understanding.concept_seeds
    assert "dynamic programming" in understanding.query_variants["bm25"].lower()
    assert "dynamic programming" in understanding.query_variants["vector"].lower()


def test_query_understanding_keeps_bfs_bare_concept_expansion():
    understanding = QueryUnderstandingService().understand("BFS")

    assert {"BFS", "Queue", "Visited Array"}.issubset(set(understanding.concept_seeds))
    assert "breadth first search" in understanding.query_variants["bm25"].lower()
    assert "breadth first search" in understanding.query_variants["vector"].lower()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("lis", "LIS"),
        ("  sorting  ", "Sorting"),
        ("0/1 knapsack", "0/1 Knapsack"),
    ],
)
def test_query_understanding_uses_document_casing_for_unregistered_exact_concepts(
    query: str,
    expected: str,
):
    document = RetrievalDocument(
        id="fixture-dp",
        source="Fixture",
        source_id="dp",
        title="Fixture Dynamic Programming Problem",
        text="Dynamic programming fixture.",
        answer="Use dynamic programming.",
        concepts=("LIS", "Sorting", "0/1 Knapsack"),
        problem_type="Dynamic Programming",
    )

    understanding = QueryUnderstandingService((document,)).understand(query)

    assert expected in understanding.concept_seeds


def test_entity_linking_uses_graph_seeds_for_chinese_problem_query():
    understanding = QueryUnderstandingService().understand(
        "給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。"
    )

    linked_entities = EntityLinkingService().link(understanding)
    entity_ids = {str(entity["entityId"]) for entity in linked_entities}

    assert {
        "concept:bfs",
        "concept:queue",
        "concept:shortest-path",
        "pattern:graph-traversal",
    }.issubset(entity_ids)
    assert any(entity.get("matchedBy") == "concept_seed" for entity in linked_entities)


def test_graph_search_returns_paths_for_chinese_problem_query():
    documents = _documents()
    graph_store = _build_graph_store(documents)
    understanding = QueryUnderstandingService().understand(
        "給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。"
    )
    linked_entities = EntityLinkingService().link(understanding)

    result = GraphSearchService(documents, graph_store=graph_store).search(
        linked_entities,
        top_k=2,
    )

    assert result.candidates
    assert result.candidates[0].id == "leetcode-994"
    assert result.paths
    assert any(_path_node_ids(path)[-1] == "concept:bfs" for path in result.paths)


def test_query_understanding_promotes_bare_queue_to_concept_seed():
    result = OnlineQueryPipeline(documents=_queue_documents()).run("queue", top_k=3)
    trace = result.trace.to_mapping()
    understanding = trace["queryUnderstanding"]

    assert "Queue" in understanding["conceptSeeds"]
    assert "concept:queue" in understanding["queryVariants"]["graphSeeds"]
    assert [candidate.id for candidate in result.graph_candidates] == [
        "leetcode-1091",
        "leetcode-994",
    ]
    assert len({candidate.score for candidate in result.graph_candidates}) == 1


def test_vector_search_service_can_use_vector_store():
    documents = _documents()
    embedding_provider = DeterministicMockEmbeddingProvider(dimension=8)
    vector_store = _build_vector_store(documents, embedding_provider)
    understanding = QueryUnderstandingService().understand("BFS shortest path with queue")

    candidates = VectorSearchService(
        documents,
        embedding_provider,
        vector_store=vector_store,
    ).search(understanding, top_k=2)

    assert candidates[0].id == "leetcode-994"
    assert candidates[0].source == "vector"
    assert candidates[0].payload["storeCandidateId"] == "leetcode-994:statement:0"
    assert candidates[0].payload["documentSource"] == "LeetCode"


def test_vector_search_uses_expanded_semantic_variant_for_chinese_query():
    class RecordingEmbeddingProvider(DeterministicMockEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__(dimension=8)
            self.requested_texts: list[str] = []

        def embed_text(self, text: str) -> tuple[float, ...]:
            self.requested_texts.append(text)
            return super().embed_text(text)

    documents = _documents()
    embedding_provider = RecordingEmbeddingProvider()
    understanding = QueryUnderstandingService().understand(
        "給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。"
    )
    service = VectorSearchService(
        documents,
        embedding_provider,
    )
    embedding_provider.requested_texts.clear()

    candidates = service.search(understanding, top_k=2)

    assert candidates
    assert candidates[0].source == "vector"
    assert embedding_provider.requested_texts == [understanding.query_variants["vector"]]
    assert "unweighted graph" in understanding.query_variants["vector"]


def test_bm25_search_service_can_use_bm25_store():
    documents = _documents()
    bm25_store = _build_bm25_store(documents)
    understanding = QueryUnderstandingService().understand("BFS queue shortest path")

    candidates = BM25SearchService(documents, bm25_store=bm25_store).search(
        understanding,
        top_k=2,
    )

    assert candidates[0].id == "leetcode-994"
    assert candidates[0].source == "bm25"
    assert candidates[0].payload["storeCandidateId"] == "leetcode-994:statement:0"
    assert candidates[0].payload["answer"] == "Use BFS from all rotten oranges."


class _FakeVectorStore:
    def __init__(self, candidates: tuple[SearchCandidate, ...]) -> None:
        self._candidates = candidates
        self.requested_top_k: list[int] = []

    def search(self, query_vector, *, top_k, filters=None):
        self.requested_top_k.append(top_k)
        return self._candidates[:top_k]


class _FakeBM25Store:
    def __init__(self, candidates: tuple[SearchCandidate, ...]) -> None:
        self._candidates = candidates
        self.requested_top_k: list[int] = []
        self.requested_queries: list[str] = []

    def search(self, query, *, top_k):
        self.requested_top_k.append(top_k)
        self.requested_queries.append(str(query))
        return self._candidates[:top_k]


def _path_node_ids(path: dict[str, object]) -> list[str]:
    nodes = path.get("nodes")
    if not isinstance(nodes, list):
        return []
    ids: list[str] = []
    for node in nodes:
        if isinstance(node, dict):
            ids.append(str(node.get("id") or ""))
        else:
            ids.append(str(node))
    return ids


def _path_relation_types(path: dict[str, object]) -> list[str]:
    relations = path.get("relations")
    if not isinstance(relations, list):
        return []
    types: list[str] = []
    for relation in relations:
        if isinstance(relation, dict):
            types.append(str(relation.get("type") or ""))
        else:
            types.append(str(relation))
    return types


def _candidate_with_raw_chunks_for_context(
    *,
    metadata_common_mistakes: list[str] | None = None,
    common_mistakes_chunk: str = "",
) -> RetrievalCandidate:
    def raw_chunk(kind: str, display_text: str, score: float) -> dict[str, object]:
        return {
            "id": f"leetcode-994:{kind}:0",
            "title": "Rotting Oranges",
            "source": "vector",
            "score": score,
            "payload": {
                "storePayload": {
                    "problemId": "leetcode-994",
                    "kind": kind,
                    "displayText": display_text,
                    "searchText": f"DO_NOT_RENDER searchText raw alias for {kind}",
                    "text": f"{display_text} fallback text",
                    "documentSource": "LeetCode",
                    "sourceId": "994",
                    "title": "Rotting Oranges",
                    "problemType": "Graph Traversal",
                    "concepts": ["BFS", "Queue"],
                },
            },
        }

    raw_chunks = [
        raw_chunk("problem_card", "Problem card display: Rotting Oranges BFS grid.", 0.99),
        raw_chunk("statement", "Statement display: oranges rot level by level.", 0.72),
        raw_chunk("solution", "Solution display: start BFS from all rotten oranges.", 0.91),
    ]
    if common_mistakes_chunk:
        raw_chunks.append(raw_chunk("common_mistakes", common_mistakes_chunk, 0.98))

    metadata: dict[str, object] = {}
    if metadata_common_mistakes is not None:
        metadata["commonMistakes"] = metadata_common_mistakes

    return RetrievalCandidate(
        id="leetcode-994",
        title="Rotting Oranges",
        source="hybrid",
        score=0.97,
        text="Candidate text should not be the selected evidence.",
        concepts=("BFS", "Queue"),
        problem_type="Graph Traversal",
        payload={
            "answer": "Use BFS from all rotten oranges.",
            "solutionHints": ["Push all rotten oranges first."],
            "difficulty": "Medium",
            "constraints": ["1 <= m, n <= 10"],
            "documentSource": "LeetCode",
            "sourceId": "994",
            "storePayload": {"metadata": metadata},
            "rawChunks": raw_chunks,
            "rawChunksComplete": True,
        },
    )


def _graph_path_for_test(
    problem_id: str,
    concept_id: str,
    concept_label: str,
    *,
    hops: int = 2,
    score: float = 1.0,
    path_source: str = "neo4j",
) -> dict[str, object]:
    nodes: list[dict[str, object]] = [
        {"id": problem_id, "label": problem_id, "layer": "problem"},
    ]
    intermediate_count = max(hops - 1, 0)
    for index in range(intermediate_count):
        nodes.append(
            {
                "id": f"source:{problem_id}:{index}",
                "label": f"source {index}",
                "layer": "source",
            }
        )
    nodes.append({"id": concept_id, "label": concept_label, "layer": "concept"})
    relations = [
        {
            "source": str(nodes[index]["id"]),
            "target": str(nodes[index + 1]["id"]),
            "type": "REQUIRES",
            "weight": score,
        }
        for index in range(len(nodes) - 1)
    ]
    return {
        "nodes": nodes,
        "relations": relations,
        "score": score,
        "pathSource": path_source,
    }


def test_prune_graph_paths_groups_by_problem_and_caps_paths():
    prune = getattr(retrieval_pipeline, "_prune_graph_paths", None)
    assert prune is not None
    raw_paths = (
        _graph_path_for_test("leetcode-994", "concept:queue", "Queue", score=0.95),
        _graph_path_for_test("leetcode-994", "concept:bfs", "BFS", score=0.4),
        _graph_path_for_test("leetcode-994", "concept:bfs", "BFS", score=0.9),
        _graph_path_for_test("leetcode-994", "concept:shortest-path", "Shortest Path", score=0.8),
        _graph_path_for_test("leetcode-1091", "concept:queue", "Queue", score=0.7),
        _graph_path_for_test("leetcode-1091", "concept:bfs", "BFS", score=0.6),
        _graph_path_for_test("uva-10653", "concept:bfs", "BFS", score=1.0),
    )

    pruned = prune(raw_paths, ("leetcode-994", "leetcode-1091"))

    assert [_path_node_ids(path)[0] for path in pruned] == [
        "leetcode-994",
        "leetcode-994",
        "leetcode-1091",
        "leetcode-1091",
    ]
    assert [_path_node_ids(path)[-1] for path in pruned] == [
        "concept:shortest-path",
        "concept:bfs",
        "concept:bfs",
        "concept:queue",
    ]
    assert pruned[1]["score"] == 0.9


def test_prune_graph_paths_prefers_shorter_paths_before_concept_priority():
    prune = getattr(retrieval_pipeline, "_prune_graph_paths", None)
    assert prune is not None
    raw_paths = (
        _graph_path_for_test("leetcode-994", "concept:shortest-path", "Shortest Path", hops=3),
        _graph_path_for_test("leetcode-994", "concept:bfs", "BFS", hops=2),
        _graph_path_for_test("leetcode-994", "concept:queue", "Queue", hops=1),
    )

    pruned = prune(raw_paths, ("leetcode-994",))

    assert [_path_node_ids(path)[-1] for path in pruned] == [
        "concept:queue",
        "concept:bfs",
    ]


def test_online_pipeline_returns_pruned_graph_paths_and_keeps_raw_paths():
    documents = _documents()
    embedding_provider = DeterministicMockEmbeddingProvider(dimension=8)
    pipeline = OnlineQueryPipeline(
        documents=documents,
        embedding_provider=embedding_provider,
        vector_store=_build_vector_store(documents, embedding_provider),
        bm25_store=_build_bm25_store(documents),
        graph_store=_build_graph_store(documents),
    )

    result = pipeline.run("BFS shortest path with queue graph traversal", top_k=2)

    assert result.raw_graph_paths
    assert len(result.graph_paths) <= len(result.raw_graph_paths)
    assert len(result.graph_paths) <= 2 * len(result.reranked_candidates)
    assert result.graph_paths == retrieval_pipeline._prune_graph_paths(
        result.raw_graph_paths,
        tuple(candidate.id for candidate in result.reranked_candidates),
    )


def test_store_backed_bm25_filters_zero_score_candidates():
    uva = _uva_document()
    leetcode = _documents()[0]
    bm25_store = _FakeBM25Store(
        (
            SearchCandidate(
                id="leetcode-994:statement:0",
                score=0.0,
                payload=_store_payload(leetcode),
            ),
            SearchCandidate(
                id="uva-10653:statement:0",
                score=0.7,
                payload=_store_payload(uva),
            ),
        )
    )
    understanding = QueryUnderstandingService((leetcode, uva)).understand("10653")

    candidates = BM25SearchService((leetcode, uva), bm25_store=bm25_store).search(
        understanding,
        top_k=3,
    )

    assert [candidate.id for candidate in candidates] == ["uva-10653"]
    assert all(candidate.score > 0 for candidate in candidates)


def test_bm25_search_service_passes_multilingual_bm25_variant_to_store():
    documents = _documents()
    store = _FakeBM25Store(
        (
            SearchCandidate(
                id="leetcode-994:statement:0",
                score=0.9,
                payload=_store_payload(documents[0]),
            ),
        )
    )
    understanding = QueryUnderstandingService().understand(
        "給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。"
    )

    candidates = BM25SearchService(documents, bm25_store=store).search(understanding, top_k=1)

    assert candidates[0].id == "leetcode-994"
    assert store.requested_queries
    assert all(query == understanding.query_variants["bm25"] for query in store.requested_queries)


def _duplicate_chunk_candidates() -> tuple[SearchCandidate, ...]:
    first, second = _documents()
    return (
        SearchCandidate(
            id="leetcode-994:statement:0",
            score=0.90,
            payload=_store_payload(first),
        ),
        SearchCandidate(
            id="leetcode-994:answer:1",
            score=0.80,
            payload={**_store_payload(first), "kind": "answer"},
        ),
        SearchCandidate(
            id="leetcode-994:hint-1:2",
            score=0.70,
            payload={**_store_payload(first), "kind": "hint"},
        ),
        SearchCandidate(
            id="leetcode-300:statement:0",
            score=0.60,
            payload=_store_payload(second),
        ),
    )


def _interleaved_chunk_candidates() -> tuple[SearchCandidate, ...]:
    first, second = _documents()
    return (
        SearchCandidate(
            id="leetcode-994:statement:0",
            score=0.90,
            payload=_store_payload(first),
        ),
        SearchCandidate(
            id="leetcode-300:statement:0",
            score=0.80,
            payload=_store_payload(second),
        ),
        SearchCandidate(
            id="leetcode-994:answer:1",
            score=0.70,
            payload={**_store_payload(first), "kind": "answer"},
        ),
    )


def _same_problem_candidates(count: int) -> tuple[SearchCandidate, ...]:
    first, _ = _documents()
    return tuple(
        SearchCandidate(
            id=f"leetcode-994:chunk:{index}",
            score=1.0 - (index / 1000),
            payload={**_store_payload(first), "kind": "chunk"},
        )
        for index in range(count)
    )


def test_store_enriches_once_after_early_unique_success():
    documents = _documents()
    store = _FakeVectorStore(_interleaved_chunk_candidates())
    understanding = QueryUnderstandingService().understand("BFS queue")

    candidates = VectorSearchService(
        documents,
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=store,
    ).search(understanding, top_k=2)

    assert store.requested_top_k == [2, 4]
    assert [
        chunk["payload"]["storeCandidateId"]
        for chunk in candidates[0].payload["rawChunks"]
    ] == [
        "leetcode-994:statement:0",
        "leetcode-994:answer:1",
    ]


def test_store_fetch_attempts_and_windows_are_tightly_bounded():
    documents = _documents()
    understanding = QueryUnderstandingService().understand("BFS queue")
    attempt_store = _FakeVectorStore(_same_problem_candidates(120))
    cap_store = _FakeVectorStore(_same_problem_candidates(120))

    VectorSearchService(
        documents,
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=attempt_store,
    ).search(understanding, top_k=10)
    capped_candidates = VectorSearchService(
        documents,
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=cap_store,
    ).search(understanding, top_k=30)

    assert attempt_store.requested_top_k == [10, 20, 40, 80]
    assert cap_store.requested_top_k == [30, 60, 100]
    assert capped_candidates[0].payload["rawChunksComplete"] is False


def test_store_marks_raw_chunks_complete_only_after_a_short_page():
    documents = _documents()
    understanding = QueryUnderstandingService().understand("BFS queue")
    exhausted_store = _FakeVectorStore(_interleaved_chunk_candidates())
    capped_store = _FakeVectorStore(_same_problem_candidates(120))

    exhausted = VectorSearchService(
        documents,
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=exhausted_store,
    ).search(understanding, top_k=2)
    capped = VectorSearchService(
        documents,
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=capped_store,
    ).search(understanding, top_k=30)

    assert exhausted[0].payload["rawChunksComplete"] is True
    assert capped[0].payload["rawChunksComplete"] is False


def test_aggregate_problem_candidates_deep_copies_nested_chunk_snapshots():
    nested_payload = {
        "storeCandidateId": "uva-10653:answer:1",
        "storePayload": {"metadata": {"tags": ["original"]}},
    }
    chunk = RetrievalCandidate(
        id="uva-10653",
        title="Bombs! NO they are Mines!!",
        source="vector",
        score=0.40,
        payload=nested_payload,
    )

    aggregated = _aggregate_problem_candidates((chunk,), source="vector", top_k=1)
    aggregate_payload = aggregated[0].payload
    raw_chunk_payload = aggregate_payload["rawChunks"][0]["payload"]

    nested_payload["storePayload"]["metadata"]["tags"].append("mutated")
    assert aggregate_payload["storePayload"]["metadata"]["tags"] == ["original"]
    assert raw_chunk_payload["storePayload"]["metadata"]["tags"] == ["original"]

    aggregate_payload["storePayload"]["metadata"]["tags"].append("aggregate-only")
    assert raw_chunk_payload["storePayload"]["metadata"]["tags"] == ["original"]


def test_vector_store_overfetches_until_it_has_requested_unique_problems():
    documents = _documents()
    store = _FakeVectorStore(_duplicate_chunk_candidates())
    understanding = QueryUnderstandingService().understand("BFS queue")

    candidates = VectorSearchService(
        documents,
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=store,
    ).search(understanding, top_k=2)

    assert store.requested_top_k == [2, 4, 8]
    assert [candidate.id for candidate in candidates] == ["leetcode-994", "leetcode-300"]
    assert [
        chunk["payload"]["storeCandidateId"]
        for chunk in candidates[0].payload["rawChunks"]
    ] == [
        "leetcode-994:statement:0",
        "leetcode-994:answer:1",
        "leetcode-994:hint-1:2",
    ]


def test_bm25_store_overfetches_until_it_has_requested_unique_problems():
    documents = _documents()
    store = _FakeBM25Store(_duplicate_chunk_candidates())
    understanding = QueryUnderstandingService().understand("BFS queue")

    candidates = BM25SearchService(documents, bm25_store=store).search(
        understanding,
        top_k=2,
    )

    assert store.requested_top_k == [2, 4, 8]
    assert [candidate.id for candidate in candidates] == ["leetcode-994", "leetcode-300"]
    assert [
        chunk["payload"]["storeCandidateId"]
        for chunk in candidates[0].payload["rawChunks"]
    ] == [
        "leetcode-994:statement:0",
        "leetcode-994:answer:1",
        "leetcode-994:hint-1:2",
    ]


def test_vector_store_omits_non_positive_candidates():
    first, second = _documents()
    store = _FakeVectorStore(
        (
            SearchCandidate(
                id="leetcode-994:statement:0",
                score=0.50,
                payload=_store_payload(first),
            ),
            SearchCandidate(
                id="leetcode-300:statement:0",
                score=-0.25,
                payload=_store_payload(second),
            ),
        )
    )
    understanding = QueryUnderstandingService().understand("BFS queue")

    candidates = VectorSearchService(
        (first, second),
        DeterministicMockEmbeddingProvider(dimension=8),
        vector_store=store,
    ).search(understanding, top_k=2)

    assert [candidate.id for candidate in candidates] == ["leetcode-994"]


def test_store_candidate_payload_preserves_enriched_evidence_fields():
    documents = _documents()
    embedding_provider = DeterministicMockEmbeddingProvider(dimension=8)
    vector_store = _build_vector_store(documents, embedding_provider)
    bm25_store = _build_bm25_store(documents)
    understanding = QueryUnderstandingService().understand("BFS shortest path with queue")

    vector_candidate = VectorSearchService(
        documents,
        embedding_provider,
        vector_store=vector_store,
    ).search(understanding, top_k=1)[0]
    bm25_candidate = BM25SearchService(documents, bm25_store=bm25_store).search(
        understanding,
        top_k=1,
    )[0]

    for candidate in (vector_candidate, bm25_candidate):
        assert candidate.payload["answer"] == "Use BFS from all rotten oranges."
        assert candidate.payload["solutionHints"] == [
            "Push all rotten oranges first.",
            "Expand one BFS layer per minute.",
        ]
        assert candidate.payload["difficulty"] == "Medium"
        assert candidate.payload["constraints"] == ["1 <= m, n <= 10"]
        assert candidate.payload["sourceId"] == "994"
        assert candidate.payload["title"] == "Rotting Oranges"
        assert candidate.payload["problemType"] == "Graph Traversal"
        assert candidate.payload["concepts"] == ["BFS", "Queue"]


def test_graph_search_service_can_use_graph_store():
    documents = _documents()
    graph_store = _build_graph_store(documents)
    understanding = QueryUnderstandingService().understand("BFS queue graph traversal")
    linked_entities = EntityLinkingService().link(understanding)

    result = GraphSearchService(documents, graph_store=graph_store).search(
        linked_entities,
        top_k=2,
    )

    assert result.candidates[0].id == "leetcode-994"
    assert result.candidates[0].source == "graph"
    assert any(
        _path_node_ids(path) == ["leetcode-994", "source:leetcode:994", "concept:bfs"]
        and _path_relation_types(path) == ["DERIVED_FROM_SOURCE", "REQUIRES"]
        and path["pathSource"] == "neo4j"
        and path["storePath"]["nodes"] == ["leetcode-994", "concept:bfs"]
        and path["storePath"]["relations"] == ["REQUIRES"]
        for path in result.paths
    )


def test_graph_relation_vocabulary_preserves_requires_and_has_pattern():
    problem = {"id": "leetcode-994"}
    concept = {"id": "concept:bfs"}
    pattern = {"id": "pattern:graph-traversal"}

    requires = _graph_edge(problem, concept, "requires", 1.0)
    has_pattern = _graph_edge(problem, pattern, "HAS_PATTERN", 1.0)

    assert requires["type"] == "REQUIRES"
    assert "normalizedFrom" not in requires
    assert has_pattern["type"] == "HAS_PATTERN"
    assert "normalizedFrom" not in has_pattern


def test_store_graph_search_uses_concept_problem_ids_when_paths_are_absent():
    documents = _queue_documents()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_entities(
        (
            EntityRecord(
                id="concept:queue",
                name="Queue",
                type="data_structure",
                problem_ids=("leetcode-994", "leetcode-1091"),
            ),
            EntityRecord(
                id=documents[0].id,
                name=documents[0].title,
                type="problem",
            ),
            EntityRecord(
                id=documents[1].id,
                name=documents[1].title,
                type="problem",
            ),
        )
    )

    understanding = QueryUnderstandingService(documents).understand("queue")
    linked = EntityLinkingService().link(understanding)
    result = GraphSearchService(documents, graph_store=graph_store).search(
        linked,
        top_k=3,
    )

    expected_ids = ["leetcode-1091", "leetcode-994"]
    assert [candidate.id for candidate in result.candidates] == expected_ids
    assert len({candidate.score for candidate in result.candidates}) == 1

    candidate_paths = [
        path
        for path in result.paths
        if path["graphPathOperation"] == "candidate_retrieval"
    ]
    assert len(candidate_paths) == len(expected_ids)
    assert [_path_node_ids(path)[0] for path in candidate_paths] == expected_ids
    assert [path["pathSource"] for path in candidate_paths] == ["neo4j", "neo4j"]


def test_graph_relation_fallback_records_normalized_from_for_unknown_types():
    problem = {"id": "leetcode-994"}
    concept = {"id": "concept:bfs"}

    edge = _graph_edge(problem, concept, "required_by", 0.7)

    assert edge["type"] == "SIMILAR_BY_FEATURE"
    assert edge["normalizedFrom"] == "REQUIRED_BY"

    document = _uva_document()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_entities(
        (
            EntityRecord(id=document.id, name=document.title, type="problem"),
            EntityRecord(id="concept:bfs", name="BFS", type="algorithm"),
        )
    )
    graph_store.upsert_relations(
        (
            RelationRecord(
                id="concept:bfs->uva-10653",
                source_id="concept:bfs",
                target_id=document.id,
                type="REQUIRED_BY",
                weight=1.0,
            ),
        )
    )
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None

    result = GraphSearchService((document,), graph_store=graph_store).search(
        (),
        matched_problem=matched,
        top_k=3,
    )

    bfs_path = next(path for path in result.paths if _path_node_ids(path)[-1] == "concept:bfs")
    assert _path_relation_types(bfs_path) == [
        "EXPANDED_FROM_EXACT_MATCH",
        "MENTIONS_CONCEPT",
    ]
    assert bfs_path["relations"][1]["normalizedFrom"] == "REQUIRED_BY"


def test_graph_search_service_scores_partial_store_entity_matches_by_coverage():
    documents = _documents()
    graph_store = _build_graph_store(documents)
    linked_entities = (
        {"entityId": "concept:bfs", "name": "BFS"},
        {"entityId": "concept:missing", "name": "Missing"},
    )

    result = GraphSearchService(documents, graph_store=graph_store).search(
        linked_entities,
        top_k=2,
    )

    assert result.candidates[0].id == "leetcode-994"
    assert result.candidates[0].score == round(1.0 / len(linked_entities), 6)
    assert any(
        path["pathSource"] == "neo4j" and "storePath" in path
        for path in result.paths
    )


def test_online_pipeline_accepts_store_injection_and_preserves_debug_outputs():
    documents = _documents()
    embedding_provider = DeterministicMockEmbeddingProvider(dimension=8)
    pipeline = OnlineQueryPipeline(
        documents=documents,
        embedding_provider=embedding_provider,
        vector_store=_build_vector_store(documents, embedding_provider),
        bm25_store=_build_bm25_store(documents),
        graph_store=_build_graph_store(documents),
    )

    result = pipeline.run("BFS shortest path with queue", top_k=2)
    evidence = EvidenceBuilder().build(result.reranked_candidates, result.graph_paths)
    context = ContextBuilder().build(result.query_understanding, evidence)
    trace = result.trace.to_mapping()

    assert result.query_understanding.intent == "problem_search"
    assert result.vector_candidates[0].id == "leetcode-994"
    assert result.bm25_candidates[0].id == "leetcode-994"
    assert result.graph_candidates[0].id == "leetcode-994"
    assert trace["vectorCandidates"][0]["payload"]["storeCandidateId"]
    assert trace["bm25Candidates"][0]["payload"]["storeCandidateId"]
    assert trace["vectorCandidates"][0]["payload"]["chunkCount"] == 1
    assert trace["bm25Candidates"][0]["payload"]["chunkCount"] == 1
    assert trace["vectorCandidates"][0]["payload"]["rawChunks"]
    assert trace["bm25Candidates"][0]["payload"]["rawChunks"]
    assert trace["graphCandidates"]
    assert any(
        path["pathSource"] == "neo4j" and "storePath" in path
        for path in result.graph_paths
    )
    assert evidence.to_mapping()["similarProblems"]
    assert "查詢理解" in context


@pytest.mark.parametrize(
    ("mode", "expected_ids", "expected_sources"),
    [
        (
            "hybrid",
            {"vector-only", "graph-only", "bm25-only"},
            {
                "vector-only": ["vector"],
                "graph-only": ["graph"],
                "bm25-only": ["bm25"],
            },
        ),
        ("vector", {"vector-only"}, {"vector-only": ["vector"]}),
        ("graph", {"graph-only"}, {"graph-only": ["graph"]}),
    ],
)
def test_online_pipeline_mode_controls_fusion_sources_and_final_results(
    mode,
    expected_ids,
    expected_sources,
):
    documents = (
        RetrievalDocument(
            id="vector-only",
            source="Test",
            source_id="vector",
            title="Vector candidate",
            text="Embedding-only candidate.",
            answer="Vector answer.",
            concepts=("Embeddings",),
            problem_type="Similarity",
        ),
        RetrievalDocument(
            id="graph-only",
            source="Test",
            source_id="graph",
            title="Graph candidate",
            text="BFS graph candidate.",
            answer="Graph answer.",
            concepts=("BFS",),
            problem_type="Graph Traversal",
        ),
        RetrievalDocument(
            id="bm25-only",
            source="Test",
            source_id="bm25",
            title="BM25 candidate",
            text="Lexical-only candidate.",
            answer="BM25 answer.",
            concepts=("Lexical Search",),
            problem_type="Search",
        ),
    )
    vector_store = _FakeVectorStore(
        (
            SearchCandidate(
                id="vector-only:statement:0",
                score=0.9,
                payload=_store_payload(documents[0]),
            ),
        )
    )
    bm25_store = _FakeBM25Store(
        (
            SearchCandidate(
                id="bm25-only:statement:0",
                score=0.8,
                payload=_store_payload(documents[2]),
            ),
        )
    )

    result = OnlineQueryPipeline(
        documents=documents,
        vector_store=vector_store,
        bm25_store=bm25_store,
    ).run("BFS", mode=mode, top_k=3)
    trace = result.trace.to_mapping()

    assert {candidate.id for candidate in result.fused_candidates} == expected_ids
    assert {candidate.id for candidate in result.reranked_candidates} == expected_ids
    assert {
        candidate.id: candidate.payload["sources"]
        for candidate in result.fused_candidates
    } == expected_sources
    assert trace["vectorCandidates"]
    assert trace["graphCandidates"]
    assert trace["bm25Candidates"]
    assert {candidate["id"] for candidate in trace["fusionScores"]} == expected_ids
    assert {candidate["id"] for candidate in trace["rerankerScores"]} == expected_ids
    if "graph-only" in expected_ids:
        assert result.graph_paths
    else:
        assert result.graph_paths == ()
    assert result.raw_graph_paths


def test_online_pipeline_top_k_limits_mode_specific_final_results():
    result = OnlineQueryPipeline(documents=_documents()).run(
        "BFS shortest path with queue",
        mode="vector",
        top_k=1,
    )

    assert len(result.fused_candidates) <= 2
    assert len(result.reranked_candidates) == 1
    assert len(EvidenceBuilder().build(result.reranked_candidates, ()).similar_problems) == 1


def test_exact_graph_paths_without_graph_candidates_are_labeled_as_exact_expansion():
    similar_document = RetrievalDocument(
        id="leetcode-1091",
        source="LeetCode",
        source_id="1091",
        title="Shortest Path in Binary Matrix",
        text="Use BFS to find a shortest path in an unweighted binary matrix.",
        answer="Run BFS over eight directions.",
        concepts=("BFS", "Queue", "Visited Array"),
        problem_type="Graph Traversal",
    )
    result = OnlineQueryPipeline(
        documents=(_uva_document(), similar_document),
        vector_store=_FakeVectorStore(
            (
                SearchCandidate(
                    id="leetcode-1091:statement:0",
                    score=0.9,
                    payload=_store_payload(similar_document),
                ),
            )
        ),
        bm25_store=_FakeBM25Store(()),
    ).run(
        "UVA-10653 - Bombs! NO they are Mines!!",
        mode="vector",
        top_k=1,
    )

    assert result.matched_problem is not None
    assert result.matched_problem.problem_id == "uva-10653"
    assert result.graph_candidates == ()
    assert result.graph_paths
    assert [candidate.id for candidate in result.reranked_candidates] == ["leetcode-1091"]

    allowed_layers = {"problem", "chunk", "concept", "code_feature", "pattern", "source"}
    allowed_relation_types = {
        "HAS_SECTION",
        "DERIVED_FROM_SOURCE",
        "MENTIONS_CONCEPT",
        "HAS_CODE_FEATURE",
        "USES_DATA_STRUCTURE",
        "IMPLEMENTS_PATTERN",
        "SIMILAR_BY_FEATURE",
        "EXPANDED_FROM_EXACT_MATCH",
    }
    required_components = {
        "minEdgeWeight",
        "meanEdgeWeight",
        "sourceBonus",
        "featureOverlap",
        "pathLengthPenalty",
    }
    for path in result.graph_paths:
        assert {node["layer"] for node in path["nodes"]} <= allowed_layers
        assert {relation["type"] for relation in path["relations"]} <= allowed_relation_types
        assert all(0 <= relation["weight"] <= 1 for relation in path["relations"])
        assert path["graphPathOperation"] == "exact_expansion"
        assert path["pathSource"] in {"inferred", "neo4j"}
        assert path["pathScoring"]["strategy"] == "weighted_layered_path_v1"
        assert path["score"] == path["pathScoring"]["score"]
        assert required_components <= set(path["pathScoring"]["components"])
        assert path["pathScoring"]["components"]["sourceBonus"] == 1.0
        assert path["pathScoring"]["components"]["featureOverlap"] == 0.0
        assert path["pathScoring"]["components"]["pathLengthPenalty"] == 0.0


def test_store_chunk_candidates_are_aggregated_by_problem_and_keep_raw_chunks():
    chunk_one = RetrievalCandidate(
        id="uva-10653",
        title="Bombs! NO they are Mines!!",
        source="vector",
        score=0.20,
        text="BFS queue",
        concepts=("BFS", "Queue"),
        problem_type="Graph Traversal",
        payload={"storeCandidateId": "uva-10653:answer:1", "kind": "answer"},
    )
    chunk_two = RetrievalCandidate(
        id="uva-10653",
        title="Bombs! NO they are Mines!!",
        source="vector",
        score=0.40,
        text="visited grid",
        concepts=("BFS", "Visited Array"),
        problem_type="Graph Traversal",
        payload={"storeCandidateId": "uva-10653:hint-1:3", "kind": "hint"},
    )
    other = RetrievalCandidate(
        id="leetcode-1091",
        title="Shortest Path in Binary Matrix",
        source="vector",
        score=0.30,
        text="shortest path",
        concepts=("BFS",),
        problem_type="Graph Traversal",
        payload={"storeCandidateId": "leetcode-1091:statement:0"},
    )

    aggregated = _aggregate_problem_candidates((chunk_one, chunk_two, other), source="vector", top_k=5)

    assert [candidate.id for candidate in aggregated] == ["uva-10653", "leetcode-1091"]
    assert aggregated[0].score == 0.40
    assert aggregated[0].payload["chunkCount"] == 2
    assert [chunk["payload"]["storeCandidateId"] for chunk in aggregated[0].payload["rawChunks"]] == [
        "uva-10653:hint-1:3",
        "uva-10653:answer:1",
    ]


def test_vector_graph_and_bm25_search_return_candidates():
    documents = _documents()
    embedding_provider = DeterministicMockEmbeddingProvider(dimension=8)
    understanding = QueryUnderstandingService().understand("BFS shortest path with queue")
    linked_entities = EntityLinkingService().link(understanding)

    vector_candidates = VectorSearchService(documents, embedding_provider).search(
        understanding,
        top_k=2,
    )
    bm25_candidates = BM25SearchService(documents).search(understanding, top_k=2)
    graph_result = GraphSearchService(documents).search(linked_entities, top_k=2)

    assert vector_candidates
    assert all(candidate.source == "vector" for candidate in vector_candidates)
    assert bm25_candidates[0].id == "leetcode-994"
    assert bm25_candidates[0].source == "bm25"
    assert graph_result.candidates[0].id == "leetcode-994"
    assert _path_node_ids(graph_result.paths[0]) == [
        "leetcode-994",
        "source:leetcode:994",
        "concept:bfs",
    ]
    assert _path_relation_types(graph_result.paths[0]) == [
        "DERIVED_FROM_SOURCE",
        "MENTIONS_CONCEPT",
    ]
    assert graph_result.paths[0]["pathSource"] == "inferred"


@pytest.mark.parametrize(
    "query",
    (
        "10653",
        "uva-10653",
        "uva 10653",
        "Bombs! NO they are Mines!!",
        "UVa Bombs! NO they are Mines!!",
    ),
)
def test_local_bm25_scores_exact_problem_aliases(query: str):
    documents = (_documents()[0], _uva_document())
    understanding = QueryUnderstandingService(documents).understand(query)

    candidates = BM25SearchService(documents).search(understanding, top_k=3)

    assert [candidate.id for candidate in candidates] == ["uva-10653"]
    assert candidates[0].score > 0
    assert candidates[0].source == "bm25"


def test_local_bm25_filters_zero_score_candidates():
    documents = (_documents()[0], _uva_document())
    understanding = QueryUnderstandingService(documents).understand("10653")

    candidates = BM25SearchService(documents).search(understanding, top_k=3)

    assert [candidate.id for candidate in candidates] == ["uva-10653"]
    assert all(candidate.score > 0 for candidate in candidates)


def test_local_bm25_returns_candidates_for_chinese_problem_query():
    documents = _documents()
    understanding = QueryUnderstandingService().understand(
        "給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。"
    )

    candidates = BM25SearchService(documents).search(understanding, top_k=2)

    assert candidates
    assert candidates[0].id == "leetcode-994"
    assert all(candidate.score > 0 for candidate in candidates)


def test_hybrid_fusion_dedupes_and_normalizes_scores_then_reranks():
    candidates = HybridFusionService().fuse(
        vector_candidates=(
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="vector",
                score=0.4,
                text="BFS queue",
                concepts=("BFS", "Queue"),
            ),
        ),
        graph_candidates=(
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="graph",
                score=1.0,
                text="BFS queue",
                concepts=("BFS", "Queue"),
            ),
        ),
        bm25_candidates=(
            RetrievalCandidate(
                id="leetcode-300",
                title="Longest Increasing Subsequence",
                source="bm25",
                score=3.0,
                text="dynamic programming",
                concepts=("Dynamic Programming",),
            ),
        ),
        top_k=3,
    )

    assert [candidate.id for candidate in candidates] == ["leetcode-994", "leetcode-300"]
    assert candidates[0].score <= 1.0
    assert candidates[0].payload["sources"] == ["graph", "vector"]

    reranked = Reranker().rerank("BFS queue", candidates, top_k=2)
    assert reranked[0].id == "leetcode-994"
    assert reranked[0].payload["rerankerScore"] >= reranked[1].payload["rerankerScore"]


def test_hybrid_fusion_counts_each_source_once_per_problem():
    candidates = HybridFusionService().fuse(
        vector_candidates=(
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="vector",
                score=0.9,
                text="BFS queue chunk one",
                concepts=("BFS", "Queue"),
            ),
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="vector",
                score=0.8,
                text="BFS queue chunk two",
                concepts=("BFS", "Queue"),
            ),
            RetrievalCandidate(
                id="leetcode-300",
                title="Longest Increasing Subsequence",
                source="vector",
                score=0.6,
                text="dynamic programming",
                concepts=("Dynamic Programming",),
            ),
        ),
        graph_candidates=(),
        bm25_candidates=(),
        top_k=2,
    )

    by_id = {candidate.id: candidate for candidate in candidates}

    assert by_id["leetcode-994"].score == 0.35
    assert by_id["leetcode-994"].payload["sources"] == ["vector"]
    assert by_id["leetcode-300"].score == round(0.35 * (0.6 / 0.9), 6)


def test_fusion_preserves_raw_chunks_from_every_chunk_source():
    vector_chunk = {
        "id": "uva-10653:hint:1",
        "title": "Bombs! NO they are Mines!!",
        "source": "vector",
        "score": 0.9,
        "payload": {"storeCandidateId": "uva-10653:hint:1", "kind": "hint"},
    }
    bm25_chunk = {
        "id": "uva-10653:statement:0",
        "title": "Bombs! NO they are Mines!!",
        "source": "bm25",
        "score": 3.0,
        "payload": {"storeCandidateId": "uva-10653:statement:0", "kind": "statement"},
    }

    candidates = HybridFusionService().fuse(
        vector_candidates=(
            RetrievalCandidate(
                id="uva-10653",
                title="Bombs! NO they are Mines!!",
                source="vector",
                score=0.9,
                text="BFS hint",
                concepts=("BFS", "Queue"),
                problem_type="Graph Traversal",
                payload={
                    "rawChunks": [vector_chunk],
                    "rawChunksComplete": True,
                },
            ),
        ),
        graph_candidates=(),
        bm25_candidates=(
            RetrievalCandidate(
                id="uva-10653",
                title="Bombs! NO they are Mines!!",
                source="bm25",
                score=3.0,
                text="BFS statement",
                concepts=("BFS", "Queue"),
                problem_type="Graph Traversal",
                payload={
                    "rawChunks": [bm25_chunk],
                    "rawChunksComplete": True,
                },
            ),
        ),
        top_k=1,
    )

    payload = candidates[0].payload
    raw_chunk_ids = [
        chunk["payload"]["storeCandidateId"]
        for chunk in payload["rawChunks"]
    ]

    assert payload["sources"] == ["vector", "bm25"]
    assert raw_chunk_ids == ["uva-10653:hint:1", "uva-10653:statement:0"]
    assert payload["chunkEvidence"] == {
        "available": True,
        "complete": True,
        "missingSources": [],
        "unavailableReason": "",
    }


def test_fusion_treats_candidate_mapping_fallback_chunks_as_available_by_default():
    candidates = HybridFusionService().fuse(
        vector_candidates=(
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="vector",
                score=0.9,
                text="BFS grid",
                concepts=("BFS",),
                problem_type="Graph Traversal",
                payload={},
            ),
        ),
        graph_candidates=(),
        bm25_candidates=(),
        top_k=1,
    )

    payload = candidates[0].payload

    assert payload["rawChunksComplete"] is True
    assert payload["chunkEvidence"] == {
        "available": True,
        "complete": True,
        "missingSources": [],
        "unavailableReason": "",
    }
    assert payload["rawChunks"][0]["source"] == "vector"


def test_hybrid_fusion_ignores_non_positive_candidates_from_every_source():
    candidates = HybridFusionService().fuse(
        vector_candidates=(
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="vector",
                score=0.8,
                text="BFS queue",
                concepts=("BFS", "Queue"),
            ),
            RetrievalCandidate(
                id="uva-10653",
                title="Bombs! NO they are Mines!!",
                source="vector",
                score=0.0,
                text="untrusted adapter row",
                concepts=("BFS",),
            ),
        ),
        graph_candidates=(
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="graph",
                score=-0.1,
                text="bad graph row",
                concepts=("BFS",),
            ),
        ),
        bm25_candidates=(
            RetrievalCandidate(
                id="leetcode-994",
                title="Rotting Oranges",
                source="bm25",
                score=0.0,
                text="bad BM25 row",
                concepts=("BFS",),
            ),
        ),
        top_k=3,
    )

    assert [candidate.id for candidate in candidates] == ["leetcode-994"]
    assert candidates[0].score == 0.35
    assert candidates[0].payload["sources"] == ["vector"]


def test_evidence_and_context_builders_create_stable_llm_context():
    documents = _documents()
    result = OnlineQueryPipeline(documents=documents).run(
        "BFS shortest path with queue",
        top_k=2,
    )

    evidence = EvidenceBuilder().build(result.reranked_candidates, result.graph_paths)
    context = ContextBuilder().build(result.query_understanding, evidence)

    evidence_map = evidence.to_mapping()
    assert evidence_map["similarProblems"][0]["id"] == "leetcode-994"
    assert evidence_map["graphPaths"]
    assert "BFS" in evidence_map["algorithmEvidence"]
    assert "Queue" in evidence_map["dataStructureEvidence"]
    assert "Graph Traversal" in evidence_map["patternEvidence"]
    assert "查詢理解" in context
    assert "Rotting Oranges" in context
    assert "常見錯誤" in context


def test_context_builder_includes_enriched_candidate_evidence():
    candidate = RetrievalCandidate(
        id="leetcode-994",
        title="Rotting Oranges",
        source="hybrid",
        score=0.97,
        text="Multi-source BFS with a queue on a grid.",
        concepts=("BFS", "Queue"),
        problem_type="Graph Traversal",
        payload={
            "answer": "Use BFS from all rotten oranges.",
            "solutionHints": ["Push all rotten oranges first."],
            "difficulty": "Medium",
            "constraints": ["1 <= m, n <= 10"],
            "documentSource": "LeetCode",
            "sourceId": "994",
        },
    )
    understanding = QueryUnderstandingService().understand("BFS queue shortest path")
    graph_paths = (
        {
            "nodes": [
                {"id": "leetcode-994", "label": "Rotting Oranges", "type": "problem"},
                {"id": "concept:bfs", "label": "BFS", "type": "algorithm"},
            ],
            "relations": ["REQUIRES"],
            "rationale": "linked BFS to Rotting Oranges",
        },
    )

    evidence = EvidenceBuilder().build((candidate,), graph_paths)
    context = ContextBuilder().build(understanding, evidence)

    similar_problem = evidence.to_mapping()["similarProblems"][0]
    assert similar_problem["answerHint"] == "Use BFS from all rotten oranges."
    assert similar_problem["solutionHints"] == ["Push all rotten oranges first."]
    assert similar_problem["difficulty"] == "Medium"
    assert similar_problem["constraints"] == ["1 <= m, n <= 10"]
    assert "查詢理解" in context
    assert "- 意圖: problem_search" in context
    assert "- 輸入類型: problem" in context
    assert "- 關鍵詞: bfs, queue, shortest, path" in context
    assert "命中題目\n- 無" in context
    assert "相似題" in context
    assert "答案摘要: Use BFS from all rotten oranges." in context
    assert "解題提示: Push all rotten oranges first." in context
    assert "難度: Medium" in context
    assert "限制: 1 <= m, n <= 10" in context
    assert "圖路徑" in context
    assert "Rotting Oranges -> BFS" in context
    assert "relations=REQUIRES" in context
    assert "rationale=linked BFS to Rotting Oranges" in context
    assert "演算法證據" in context
    assert "資料結構證據" in context
    assert "技巧證據" in context
    assert "題型證據" in context
    assert "常見錯誤" in context


def test_context_builder_ignores_search_text_noise_from_store_payload():
    alias_spam = "ALIAS_SPAM bfs breadth-first-search queue alias spam repeated"
    candidate = RetrievalCandidate(
        id="leetcode-994",
        title="Rotting Oranges",
        source="hybrid",
        score=0.97,
        text="Multi-source BFS with a queue on a grid.",
        concepts=("BFS", "Queue"),
        problem_type="Graph Traversal",
        payload={
            "text": "Multi-source BFS with a queue on a grid.",
            "displayText": "Multi-source BFS with a queue on a grid.",
            "searchText": f"Multi-source BFS with a queue on a grid. {alias_spam} {alias_spam}",
            "answer": "Use BFS from all rotten oranges.",
            "solutionHints": ["Push all rotten oranges first."],
            "difficulty": "Medium",
            "constraints": ["1 <= m, n <= 10"],
            "documentSource": "LeetCode",
            "sourceId": "994",
        },
    )
    understanding = QueryUnderstandingService().understand("BFS queue shortest path")

    evidence = EvidenceBuilder().build((candidate,), ())
    context = ContextBuilder().build(understanding, evidence)

    assert "Use BFS from all rotten oranges." in context
    assert "Push all rotten oranges first." in context
    assert alias_spam not in context


def _dp_retrieval_candidate() -> RetrievalCandidate:
    return RetrievalCandidate(
        id="uva-437",
        title="The Tower of Babylon",
        source="reranker",
        score=0.91,
        text="Solve tower stacking with dynamic programming over sorted block orientations.",
        concepts=("DP", "LIS", "Sorting"),
        problem_type="Dynamic Programming",
        payload={
            "documentSource": "UVa",
            "sourceId": "437",
            "answer": "Generate all block orientations, sort them, then run DP for the tallest stack.",
            "solutionHints": [
                "Create three orientations for each block.",
                "Sort base dimensions before applying LIS-style DP.",
            ],
            "difficulty": "Medium",
            "provenance": {"source": "seed"},
            "rawChunks": [
                {
                    "id": "uva-437:solution:0",
                    "kind": "solution",
                    "displayText": "Tower DP solution chunk.",
                    "score": 0.88,
                }
            ],
            "chunkEvidence": {"available": True, "complete": True},
            "chunkCount": 1,
            "rawChunksComplete": True,
        },
    )


def _bfs_retrieval_candidate() -> RetrievalCandidate:
    return RetrievalCandidate(
        id="leetcode-1091",
        title="Shortest Path in Binary Matrix",
        source="reranker",
        score=0.93,
        text="Use BFS with a queue over eight grid directions.",
        concepts=("BFS", "Queue", "Visited Array"),
        problem_type="Graph Traversal",
        payload={
            "answer": "Run BFS from the start cell to find the shortest path.",
            "solutionHints": ["Use a queue and mark each visited cell once."],
        },
    )


def test_evidence_builder_exact_match_uses_matched_problem_scope_before_unrelated_candidates():
    dp = _dp_retrieval_candidate()
    bfs = _bfs_retrieval_candidate()
    matched = ExactProblemMatch(
        problem_id="uva-437",
        title="The Tower of Babylon",
        source="UVa",
        source_id="437",
        match_kind="exact_problem_id",
        confidence=1.0,
        candidate=dp,
    )

    evidence = EvidenceBuilder().build(
        (bfs, dp),
        (),
        matched_problem=matched,
        query_concepts=("Dynamic Programming",),
    ).to_mapping()

    assert evidence["matchedProblem"]["id"] == "uva-437"
    assert evidence["similarProblems"] == []
    assert "Dynamic Programming" in evidence["algorithmEvidence"]
    assert "BFS" not in evidence["algorithmEvidence"]
    assert "Binary Search" not in evidence["algorithmEvidence"]
    assert "Queue" not in evidence["dataStructureEvidence"]
    assert "Stack" not in evidence["dataStructureEvidence"]
    assert "Graph Traversal" not in evidence["patternEvidence"]
    mistakes = " ".join(evidence["commonMistakes"]).lower()
    assert all(term not in mistakes for term in ("queue", "binary search", "stack"))


def test_evidence_builder_exact_match_displays_only_candidates_with_contained_scope():
    matched_candidate = _dp_retrieval_candidate()
    out_of_scope_candidate = RetrievalCandidate(
        id="uva-10130",
        title="SuperSale",
        source="reranker",
        score=0.89,
        text="Solve a shopping problem with dynamic programming.",
        concepts=("DP", "0/1 Knapsack"),
        problem_type="Dynamic Programming",
        payload={
            "answer": "Use a capacity-indexed table.",
            "commonMistakes": ["SECONDARY_ONLY_MISTAKE"],
        },
    )
    display_candidate = RetrievalCandidate(
        id="fixture-contained-dp",
        title="Contained Dynamic Programming Fixture",
        source="reranker",
        score=0.88,
        text="Solve a sorted subsequence problem with dynamic programming.",
        concepts=("DP", "LIS"),
        problem_type="Dynamic Programming",
        payload={"answer": "Use LIS-style dynamic programming."},
    )
    matched = ExactProblemMatch(
        problem_id=matched_candidate.id,
        title=matched_candidate.title,
        source="UVa",
        source_id="437",
        match_kind="exact_problem_id",
        confidence=1.0,
        candidate=matched_candidate,
    )

    evidence = EvidenceBuilder().build(
        (out_of_scope_candidate, display_candidate, matched_candidate),
        (),
        matched_problem=matched,
        query_concepts=("BFS", "Queue", "Visited Array"),
    ).to_mapping()

    assert [problem["id"] for problem in evidence["similarProblems"]] == [
        "fixture-contained-dp"
    ]
    assert evidence["similarProblems"][0]["sharedConcepts"] == [
        "Dynamic Programming",
        "LIS",
    ]
    assert evidence["matchedProblem"]["sharedConcepts"] == [
        "Dynamic Programming",
        "LIS",
        "Sorting",
    ]
    assert evidence["algorithmEvidence"] == ["Dynamic Programming"]
    assert evidence["patternEvidence"] == ["Dynamic Programming"]
    assert "0/1 Knapsack" not in evidence["algorithmEvidence"]
    assert "SECONDARY_ONLY_MISTAKE" not in evidence["commonMistakes"]


def test_evidence_builder_non_exact_concept_uses_first_candidate_for_evidence_only():
    first = replace(
        _dp_retrieval_candidate(),
        payload={
            **_dp_retrieval_candidate().payload,
            "commonMistakes": ["FIRST_CANDIDATE_MISTAKE"],
        },
    )
    second = RetrievalCandidate(
        id="uva-10130",
        title="SuperSale",
        source="reranker",
        score=0.87,
        text="Use dynamic programming for repeated knapsack queries.",
        concepts=("DP", "0/1 Knapsack"),
        problem_type="Dynamic Programming",
        payload={
            "answer": "Use 0/1 knapsack.",
            "commonMistakes": ["SECOND_CANDIDATE_MISTAKE"],
        },
    )

    evidence = EvidenceBuilder().build(
        (first, second),
        (),
        query_concepts=("Dynamic Programming",),
    ).to_mapping()

    assert [problem["id"] for problem in evidence["similarProblems"]] == [
        "uva-437",
        "uva-10130",
    ]
    assert evidence["commonMistakes"] == ["FIRST_CANDIDATE_MISTAKE"]


def test_evidence_builder_filters_graph_paths_by_selected_ids_and_active_concepts():
    matched_candidate = _dp_retrieval_candidate()
    display_candidate = RetrievalCandidate(
        id="uva-10130",
        title="SuperSale",
        source="reranker",
        score=0.87,
        text="Dynamic programming fixture.",
        concepts=("DP", "LIS"),
        problem_type="Dynamic Programming",
    )
    excluded_candidate = _bfs_retrieval_candidate()
    matched = ExactProblemMatch(
        problem_id=matched_candidate.id,
        title=matched_candidate.title,
        source="UVa",
        source_id="437",
        match_kind="exact_problem_id",
        confidence=1.0,
        candidate=matched_candidate,
    )

    def path(problem_id: str, concept_id: str, concept_label: str) -> dict[str, object]:
        return {
            "nodes": [
                {"id": problem_id, "label": problem_id, "type": "problem"},
                {"id": concept_id, "label": concept_label, "type": "concept"},
            ],
            "relations": [{"type": "REQUIRES"}],
            "score": 0.9,
            "rationale": f"{problem_id} uses {concept_label}",
        }

    evidence = EvidenceBuilder().build(
        (excluded_candidate, display_candidate, matched_candidate),
        (
            path(matched_candidate.id, "concept:dynamic-programming", "Dynamic Programming"),
            path(display_candidate.id, "concept:dynamic-programming", "Dynamic Programming"),
            path(display_candidate.id, "concept:bfs", "BFS"),
            path(excluded_candidate.id, "concept:bfs", "BFS"),
        ),
        matched_problem=matched,
        query_concepts=("BFS",),
    ).to_mapping()

    assert [
        (_path_node_ids(graph_path)[0], _path_node_ids(graph_path)[-1])
        for graph_path in evidence["graphPaths"]
    ] == [
        ("uva-437", "concept:dynamic-programming"),
        ("uva-10130", "concept:dynamic-programming"),
    ]


def test_evidence_builder_strips_store_path_with_out_of_scope_intermediate_concept():
    evidence = EvidenceBuilder().build(
        (_dp_retrieval_candidate(),),
        (
            {
                "nodes": [
                    {"id": "uva-437", "label": "uva-437", "type": "problem"},
                    {
                        "id": "concept:dynamic-programming",
                        "label": "Dynamic Programming",
                        "type": "concept",
                    },
                ],
                "relations": [{"type": "REQUIRES"}],
                "storePath": {
                    "nodes": [
                        "uva-437",
                        "concept:bfs",
                        "concept:dynamic-programming",
                    ],
                    "relations": ["REQUIRES", "RELATED_TO"],
                },
            },
        ),
        query_concepts=("Dynamic Programming",),
    ).to_mapping()

    graph_path, = evidence["graphPaths"]
    assert "storePath" not in graph_path
    assert "bfs" not in json.dumps(graph_path).lower()


def test_evidence_builder_without_exact_match_uses_first_query_consistent_candidate():
    dp = _dp_retrieval_candidate()
    bfs = _bfs_retrieval_candidate()

    evidence = EvidenceBuilder().build(
        (bfs, dp),
        (),
        query_concepts=("Dynamic Programming",),
    ).to_mapping()

    assert [problem["id"] for problem in evidence["similarProblems"]] == ["uva-437"]
    similar_problem = evidence["similarProblems"][0]
    assert similar_problem["title"] == "The Tower of Babylon"
    assert similar_problem["source"] == "UVa"
    assert similar_problem["sourceId"] == "437"
    assert similar_problem["answerHint"] == (
        "Generate all block orientations, sort them, then run DP for the tallest stack."
    )
    assert similar_problem["solutionHints"] == [
        "Create three orientations for each block.",
        "Sort base dimensions before applying LIS-style DP.",
    ]
    assert similar_problem["provenance"] == {"source": "seed"}
    assert similar_problem["chunkCount"] == 1
    assert similar_problem["rawChunksComplete"] is True
    assert similar_problem["chunkEvidence"] == {"available": True, "complete": True}
    assert similar_problem["matchedChunk"] == {
        "id": "uva-437:solution:0",
        "kind": "solution",
        "displayText": "Tower DP solution chunk.",
        "score": 0.88,
    }
    assert "Dynamic Programming" in evidence["algorithmEvidence"]
    assert "BFS" not in evidence["algorithmEvidence"]
    assert "Queue" not in evidence["dataStructureEvidence"]
    assert "Graph Traversal" not in evidence["patternEvidence"]


def test_evidence_builder_sanitizes_similar_problem_raw_chunk_payloads():
    candidate = RetrievalCandidate(
        id="fixture-problem",
        title="Fixture Problem",
        source="hybrid",
        score=0.91,
        text="Fixture retrieval candidate.",
        concepts=("BFS",),
        problem_type="Graph Traversal",
        payload={
            "documentSource": "FixtureSource",
            "sourceId": "fixture-1",
            "rawChunks": [
                {
                    "id": "fixture-problem:solution:0",
                    "source": "vector",
                    "score": 0.82,
                    "payload": {
                        "storeCandidateId": "fixture-problem:solution:0",
                        "kind": "solution",
                        "displayText": "Safe displayed chunk text.",
                        "documentSource": "FixtureSource",
                        "sourceId": "fixture-1",
                        "title": "Fixture Problem",
                        "problemType": "Graph Traversal",
                        "concepts": ["BFS"],
                        "metadata": {"source": "fixture", "notes": "hidden notes"},
                        "provenance": {"source": "fixture"},
                        "promptContext": "hidden prompt context",
                        "answer": "hidden answer",
                        "rawAnswer": "hidden raw answer",
                        "explanation": "hidden explanation",
                        "notes": "hidden payload notes",
                        "searchText": "index-only search text",
                        "text": "fallback free text",
                    },
                }
            ],
            "chunkCount": 1,
            "rawChunksComplete": True,
        },
    )

    evidence = EvidenceBuilder().build((candidate,), ()).to_mapping()

    raw_chunk = evidence["similarProblems"][0]["rawChunks"][0]
    assert raw_chunk["id"] == "fixture-problem:solution:0"
    assert raw_chunk["source"] == "vector"
    assert raw_chunk["score"] == 0.82
    assert raw_chunk["payload"]["storeCandidateId"] == "fixture-problem:solution:0"
    assert raw_chunk["payload"]["kind"] == "solution"
    assert raw_chunk["payload"]["displayText"] == "Safe displayed chunk text."
    assert raw_chunk["payload"]["metadata"] == {"source": "fixture"}
    assert raw_chunk["payload"]["provenance"] == {"source": "fixture"}

    raw_chunk_text = json.dumps(raw_chunk, ensure_ascii=False)
    for leaked_text in (
        "hidden prompt context",
        "hidden answer",
        "hidden raw answer",
        "hidden explanation",
        "hidden notes",
        "hidden payload notes",
        "index-only search text",
        "fallback free text",
    ):
        assert leaked_text not in raw_chunk_text


def test_evidence_builder_sanitizes_provenance_collections_at_every_level():
    candidate = replace(
        _dp_retrieval_candidate(),
        payload={
            **_dp_retrieval_candidate().payload,
            "provenance": [
                "SCALAR_PROVENANCE_POISON",
                {
                    "source": "seed",
                    "sourceId": "437",
                    "notes": "NESTED_PROVENANCE_POISON",
                    "metadata": {
                        "source": "seed-metadata",
                        "displayText": "PROVENANCE_METADATA_POISON",
                    },
                },
            ],
            "rawChunks": [
                {
                    "id": "uva-437:solution:0",
                    "source": "vector",
                    "score": 0.88,
                    "payload": {
                        "kind": "solution",
                        "displayText": "Safe displayed chunk text.",
                        "provenance": [
                            "RAW_CHUNK_SCALAR_POISON",
                            {
                                "source": "store",
                                "sourceId": "437",
                                "notes": "RAW_CHUNK_NESTED_POISON",
                                "metadata": {
                                    "source": "store-metadata",
                                    "displayText": "RAW_CHUNK_METADATA_POISON",
                                },
                            },
                        ],
                    },
                }
            ],
        },
    )

    evidence = EvidenceBuilder().build(
        (candidate,),
        (),
        query_concepts=("Dynamic Programming",),
    ).to_mapping()
    similar = evidence["similarProblems"][0]

    assert similar["provenance"] == [
        {
            "source": "seed",
            "sourceId": "437",
            "metadata": {"source": "seed-metadata"},
        }
    ]
    assert similar["rawChunks"][0]["payload"]["provenance"] == [
        {
            "source": "store",
            "sourceId": "437",
            "metadata": {"source": "store-metadata"},
        }
    ]
    assert similar["rawChunks"][0]["payload"]["displayText"] == (
        "Safe displayed chunk text."
    )
    serialized = json.dumps(similar, ensure_ascii=False)
    assert "POISON" not in serialized


def test_evidence_builder_selects_matched_problem_display_evidence():
    candidate = _candidate_with_raw_chunks_for_context(
        metadata_common_mistakes=["Mark each orange once."]
    )
    matched = ExactProblemMatch(
        problem_id="leetcode-994",
        title="Rotting Oranges",
        source="LeetCode",
        source_id="994",
        match_kind="exact_problem_id",
        confidence=1.0,
        candidate=candidate,
    )

    evidence = EvidenceBuilder().build((candidate,), (), matched_problem=matched)
    matched_problem = evidence.to_mapping()["matchedProblem"]

    assert matched_problem["id"] == "leetcode-994"
    assert matched_problem["solutionHints"] == ["Push all rotten oranges first."]
    assert matched_problem["problemCard"] == {
        "id": "leetcode-994:problem_card:0",
        "kind": "problem_card",
        "displayText": "Problem card display: Rotting Oranges BFS grid.",
        "score": 0.99,
    }
    assert matched_problem["statement"] == {
        "id": "leetcode-994:statement:0",
        "kind": "statement",
        "displayText": "Statement display: oranges rot level by level.",
        "score": 0.72,
    }
    assert matched_problem["solution"] == {
        "id": "leetcode-994:solution:0",
        "kind": "solution",
        "displayText": "Solution display: start BFS from all rotten oranges.",
        "score": 0.91,
    }


def test_evidence_builder_selects_similar_problem_card_and_best_display_chunk():
    candidate = _candidate_with_raw_chunks_for_context(
        common_mistakes_chunk="- Do not mutate fresh oranges twice."
    )

    evidence = EvidenceBuilder().build((candidate,), ())
    similar_problem = evidence.to_mapping()["similarProblems"][0]

    assert similar_problem["problemCard"] == {
        "id": "leetcode-994:problem_card:0",
        "kind": "problem_card",
        "displayText": "Problem card display: Rotting Oranges BFS grid.",
        "score": 0.99,
    }
    assert similar_problem["matchedChunk"] == {
        "id": "leetcode-994:solution:0",
        "kind": "solution",
        "displayText": "Solution display: start BFS from all rotten oranges.",
        "score": 0.91,
    }


def test_context_builder_uses_display_text_not_search_text():
    candidate = _candidate_with_raw_chunks_for_context(
        metadata_common_mistakes=["Mark each orange once."]
    )
    understanding = QueryUnderstandingService().understand("BFS queue shortest path")

    evidence = EvidenceBuilder().build((candidate,), ())
    context = ContextBuilder().build(understanding, evidence)

    assert "Problem card display: Rotting Oranges BFS grid." in context
    assert "Solution display: start BFS from all rotten oranges." in context
    assert "Mark each orange once." in context
    assert "DO_NOT_RENDER" not in context
    assert "searchText" not in context
    assert "rawChunks" not in context
    assert "storePayload" not in context


def test_evidence_builder_reads_common_mistakes_from_metadata():
    candidate = _candidate_with_raw_chunks_for_context(
        metadata_common_mistakes=[
            "Do not enqueue the same cell twice.",
            "Do not forget the minute boundary.",
        ],
        common_mistakes_chunk="- Chunk value should be lower priority.",
    )

    evidence = EvidenceBuilder().build((candidate,), ())

    assert evidence.to_mapping()["commonMistakes"] == [
        "Do not enqueue the same cell twice.",
        "Do not forget the minute boundary.",
        "Chunk value should be lower priority.",
    ]


def test_evidence_builder_reads_common_mistakes_from_chunk_before_template():
    candidate = _candidate_with_raw_chunks_for_context(
        common_mistakes_chunk=(
            "- Do not spread rot without level counting.\n"
            "- Do not ignore initially empty grids."
        )
    )

    evidence = EvidenceBuilder().build((candidate,), ())

    assert evidence.to_mapping()["commonMistakes"] == [
        "Do not spread rot without level counting.",
        "Do not ignore initially empty grids.",
    ]


def test_context_builder_includes_matched_problem_separately():
    matched = ExactProblemMatcher((_uva_document(),)).match(
        QueryUnderstandingService((_uva_document(),)).understand(
            "UVA-10653 - Bombs! NO they are Mines!!"
        )
    )
    assert matched is not None
    similar = RetrievalCandidate(
        id="leetcode-1091",
        title="Shortest Path in Binary Matrix",
        source="reranker",
        score=0.82,
        concepts=("BFS", "Queue"),
        problem_type="Graph Traversal",
        payload={"answer": "Run BFS over eight directions."},
    )
    evidence = EvidenceBuilder().build(
        (matched.candidate, similar),
        (),
        matched_problem=matched,
    )

    context = ContextBuilder().build(
        QueryUnderstandingService((_uva_document(),)).understand(
            "UVA-10653 - Bombs! NO they are Mines!!"
        ),
        evidence,
    )

    assert "命中題目" in context
    assert "id: uva-10653" in context
    assert "title: Bombs! NO they are Mines!!" in context
    assert "matchKind: exact_problem_id" in context
    assert "confidence: 1.0" in context
    assert "答案摘要: Run BFS from the start cell while skipping bomb cells." in context
    assert "解題提示: Mark bomb cells before BFS." in context
    assert "相似題" in context
    assert "leetcode-1091 Shortest Path in Binary Matrix" in context
    assert "Run BFS over eight directions." in context


def test_context_builder_omits_similar_problem_section_when_empty():
    matched = ExactProblemMatcher((_uva_document(),)).match(
        QueryUnderstandingService((_uva_document(),)).understand(
            "UVA-10653 - Bombs! NO they are Mines!!"
        )
    )
    assert matched is not None
    evidence = EvidenceBuilder().build(
        (matched.candidate,),
        (
            {
                "nodes": ["uva-10653", "concept:bfs"],
                "relations": ["USES"],
                "rationale": "matched problem uses BFS",
            },
        ),
        matched_problem=matched,
    )

    context = ContextBuilder().build(
        QueryUnderstandingService((_uva_document(),)).understand(
            "UVA-10653 - Bombs! NO they are Mines!!"
        ),
        evidence,
    )

    assert evidence.to_mapping()["similarProblems"] == []
    assert "命中題目" in context
    assert "id: uva-10653" in context
    assert "相似題" not in context
    assert "圖路徑" in context
    assert "uva-10653 -> concept:bfs" in context
    assert "演算法證據" in context
    assert "常見錯誤" in context


def test_online_pipeline_trace_has_required_debug_sections():
    result = OnlineQueryPipeline(documents=_documents()).run("BFS shortest path", top_k=2)

    trace = result.trace.to_mapping()
    assert trace["queryUnderstanding"]["intent"] == "problem_search"
    assert trace["entityLinking"]
    assert trace["vectorCandidates"]
    assert trace["graphCandidates"]
    assert trace["bm25Candidates"]
    assert trace["fusionScores"]
    assert trace["rerankerScores"]


def test_trace_candidates_and_graph_paths_include_score_metadata():
    result = OnlineQueryPipeline(documents=_documents()).run("BFS shortest path", top_k=2)

    trace = result.trace.to_mapping()
    expected_stages = {
        "vectorCandidates": ("vector", "Vector similarity", False),
        "graphCandidates": ("graph", "Graph match", False),
        "bm25Candidates": ("bm25", "BM25 lexical score", False),
        "fusionScores": ("fusion", "Hybrid fusion score", False),
        "rerankerScores": ("reranker", "Reranker score", False),
    }
    for lane, (stage, display_label, comparable) in expected_stages.items():
        assert trace[lane]
        for candidate in trace[lane]:
            assert candidate["scoreMeta"] == {
                "stage": stage,
                "displayLabel": display_label,
                "comparableAcrossStages": comparable,
            }

    assert result.graph_paths
    for path in result.graph_paths:
        assert path["scoreMeta"] == {
            "stage": "graph_path",
            "displayLabel": "Graph path confidence",
            "comparableAcrossStages": False,
        }


def _uva_document() -> RetrievalDocument:
    return RetrievalDocument(
        id="uva-10653",
        source="UVa",
        source_id="10653",
        title="Bombs! NO they are Mines!!",
        text="Find the shortest safe path on a grid with bomb cells.",
        answer="Run BFS from the start cell while skipping bomb cells.",
        concepts=("BFS", "Queue", "Visited Array"),
        problem_type="Graph Traversal",
        solution_hints=("Mark bomb cells before BFS.", "Track visited grid cells when enqueued."),
        difficulty="Medium",
    )


def test_graph_search_for_exact_problem_returns_problem_node_paths_with_source_labels():
    document = _uva_document()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_entities(
        (
            EntityRecord(id=document.id, name=document.title, type="problem"),
            EntityRecord(id="concept:bfs", name="BFS", type="algorithm"),
            EntityRecord(id="concept:queue", name="Queue", type="data_structure"),
        )
    )
    graph_store.upsert_relations(
        (
            RelationRecord(
                id="uva-10653->concept:bfs",
                source_id=document.id,
                target_id="concept:bfs",
                type="REQUIRES",
                weight=1.0,
            ),
            RelationRecord(
                id="uva-10653->concept:queue",
                source_id=document.id,
                target_id="concept:queue",
                type="REQUIRES",
                weight=1.0,
            ),
        )
    )
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None
    linked_entities = EntityLinkingService().link(
        understanding,
        matched_problem=matched,
    )

    result = GraphSearchService((document,), graph_store=graph_store).search(
        linked_entities,
        matched_problem=matched,
        top_k=3,
    )

    assert result.candidates == ()
    assert result.paths
    paths_by_target = {_path_node_ids(path)[-1]: path for path in result.paths}
    assert paths_by_target["concept:bfs"]["pathSource"] == "neo4j"
    assert paths_by_target["concept:queue"]["pathSource"] == "neo4j"
    assert paths_by_target["concept:visited-array"]["pathSource"] == "inferred"
    assert paths_by_target["pattern:graph-traversal"]["pathSource"] == "inferred"
    assert ["uva-10653", "source:uva:10653", "concept:bfs"] in [
        _path_node_ids(path) for path in result.paths
    ]


def test_graph_search_for_exact_problem_combines_partial_store_paths_with_inferred_missing_paths():
    document = _uva_document()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_entities(
        (
            EntityRecord(id=document.id, name=document.title, type="problem"),
            EntityRecord(id="concept:bfs", name="BFS", type="algorithm"),
        )
    )
    graph_store.upsert_relations(
        (
            RelationRecord(
                id="uva-10653->concept:bfs",
                source_id=document.id,
                target_id="concept:bfs",
                type="REQUIRES",
                weight=1.0,
            ),
        )
    )
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None

    result = GraphSearchService((document,), graph_store=graph_store).search(
        (),
        matched_problem=matched,
        top_k=3,
    )

    paths_by_target = {_path_node_ids(path)[-1]: path for path in result.paths}
    assert paths_by_target["concept:bfs"]["pathSource"] == "neo4j"
    assert paths_by_target["concept:queue"]["pathSource"] == "inferred"
    assert paths_by_target["concept:visited-array"]["pathSource"] == "inferred"
    assert paths_by_target["pattern:graph-traversal"]["pathSource"] == "inferred"
    assert paths_by_target["concept:bfs"]["storePath"]["nodes"] == [
        "uva-10653",
        "concept:bfs",
    ]


def test_graph_search_for_exact_problem_uses_reverse_store_path_with_canonical_public_path():
    document = _uva_document()
    graph_store = InMemoryGraphStore()
    graph_store.upsert_entities(
        (
            EntityRecord(id=document.id, name=document.title, type="problem"),
            EntityRecord(id="concept:bfs", name="BFS", type="algorithm"),
        )
    )
    graph_store.upsert_relations(
        (
            RelationRecord(
                id="concept:bfs->uva-10653",
                source_id="concept:bfs",
                target_id=document.id,
                type="REQUIRED_BY",
                weight=1.0,
            ),
        )
    )
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None

    result = GraphSearchService((document,), graph_store=graph_store).search(
        (),
        matched_problem=matched,
        top_k=3,
    )

    bfs_paths = [path for path in result.paths if _path_node_ids(path)[-1] == "concept:bfs"]
    assert len(bfs_paths) == 1
    assert _path_node_ids(bfs_paths[0]) == ["uva-10653", "source:uva:10653", "concept:bfs"]
    assert _path_relation_types(bfs_paths[0]) == [
        "EXPANDED_FROM_EXACT_MATCH",
        "MENTIONS_CONCEPT",
    ]
    assert bfs_paths[0]["pathSource"] == "neo4j"
    assert bfs_paths[0]["storePath"]["nodes"] == ["concept:bfs", "uva-10653"]
    assert bfs_paths[0]["storePath"]["relations"] == ["REQUIRED_BY"]


def test_graph_search_for_exact_problem_prefers_valid_reverse_store_path_over_malformed_direct_path():
    class MalformedDirectValidReverseGraphStore:
        def find_paths(self, source_id, target_id, *, max_hops=3):
            if (source_id, target_id) == ("uva-10653", "concept:bfs"):
                return (
                    {
                        "nodes": [],
                        "relations": [],
                        "score": "not-a-score",
                    },
                )
            if (source_id, target_id) == ("concept:bfs", "uva-10653"):
                return (
                    {
                        "nodes": ["concept:bfs", "uva-10653"],
                        "relations": ["REQUIRED_BY"],
                        "score": 1.0,
                    },
                )
            return ()

    document = _uva_document()
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None

    result = GraphSearchService(
        (document,),
        graph_store=MalformedDirectValidReverseGraphStore(),
    ).search(
        (),
        matched_problem=matched,
        top_k=3,
    )

    bfs_paths = [path for path in result.paths if _path_node_ids(path)[-1] == "concept:bfs"]
    assert len(bfs_paths) == 1
    assert _path_node_ids(bfs_paths[0]) == ["uva-10653", "source:uva:10653", "concept:bfs"]
    assert _path_relation_types(bfs_paths[0]) == [
        "EXPANDED_FROM_EXACT_MATCH",
        "MENTIONS_CONCEPT",
    ]
    assert bfs_paths[0]["pathSource"] == "neo4j"
    assert bfs_paths[0]["storePath"]["nodes"] == ["concept:bfs", "uva-10653"]
    assert bfs_paths[0]["storePath"]["relations"] == ["REQUIRED_BY"]


@pytest.mark.parametrize("score", ["not-a-score", None, float("nan"), float("inf"), float("-inf")])
def test_graph_search_for_exact_problem_normalizes_malformed_store_paths_without_crashing(score):
    class MalformedGraphStore:
        def find_paths(self, source_id, target_id, *, max_hops=3):
            if target_id == "concept:bfs":
                return (
                    {
                        "nodes": "not-a-node-list",
                        "relations": {"bad": "relation-container"},
                        "score": score,
                    },
                )
            return ()

    document = _uva_document()
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None

    result = GraphSearchService((document,), graph_store=MalformedGraphStore()).search(
        (),
        matched_problem=matched,
        top_k=3,
    )

    bfs_path = next(path for path in result.paths if _path_node_ids(path)[-1] == "concept:bfs")
    assert bfs_path["score"] == 0.0
    assert bfs_path["pathSource"] == "neo4j"
    assert bfs_path["storePath"] == {"nodes": [], "relations": []}


def test_graph_search_for_exact_problem_deduplicates_direct_and_reverse_store_paths():
    class DuplicateDirectionGraphStore:
        def __init__(self) -> None:
            self.calls = []

        def find_paths(self, source_id, target_id, *, max_hops=3):
            self.calls.append((source_id, target_id, max_hops))
            if (source_id, target_id) == ("uva-10653", "concept:bfs"):
                return (
                    {
                        "nodes": ["uva-10653", "concept:bfs"],
                        "relations": ["REQUIRES"],
                        "score": 1.0,
                    },
                )
            if (source_id, target_id) == ("concept:bfs", "uva-10653"):
                return (
                    {
                        "nodes": ["concept:bfs", "uva-10653"],
                        "relations": ["REQUIRED_BY"],
                        "score": 1.0,
                    },
                )
            return ()

    document = _uva_document()
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None
    graph_store = DuplicateDirectionGraphStore()

    result = GraphSearchService(
        (document,),
        graph_store=graph_store,
    ).search(
        (),
        matched_problem=matched,
        top_k=3,
    )

    assert ("concept:bfs", "uva-10653", 3) in graph_store.calls
    assert [
        _path_node_ids(path)
        for path in result.paths
        if _path_node_ids(path) == ["uva-10653", "source:uva:10653", "concept:bfs"]
    ] == [["uva-10653", "source:uva:10653", "concept:bfs"]]


def test_graph_search_for_exact_problem_skips_empty_problem_type_store_target():
    class RecordingGraphStore:
        def __init__(self) -> None:
            self.find_paths_calls: list[tuple[str, str, int]] = []

        def find_paths(self, source_id, target_id, *, max_hops=3):
            self.find_paths_calls.append((source_id, target_id, max_hops))
            return ()

    document = RetrievalDocument(
        id="uva-10653",
        source="UVa",
        source_id="10653",
        title="Bombs! NO they are Mines!!",
        text="Find the shortest safe path on a grid with bomb cells.",
        answer="Run BFS from the start cell while skipping bomb cells.",
        concepts=("BFS",),
        problem_type="",
    )
    graph_store = RecordingGraphStore()
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None

    GraphSearchService((document,), graph_store=graph_store).search(
        (),
        matched_problem=matched,
        top_k=1,
    )

    assert graph_store.find_paths_calls == [
        ("uva-10653", "concept:bfs", 3),
        ("concept:bfs", "uva-10653", 3),
    ]
    assert all(
        target_id != "pattern:unknown"
        for _, target_id, _ in graph_store.find_paths_calls
    )


def test_graph_search_marks_document_concept_fallback_paths_as_inferred():
    document = _uva_document()
    understanding = QueryUnderstandingService((document,)).understand("UVA-10653")
    matched = ExactProblemMatcher((document,)).match(understanding)
    assert matched is not None
    linked_entities = EntityLinkingService().link(
        understanding,
        matched_problem=matched,
    )

    result = GraphSearchService((document,)).search(
        linked_entities,
        matched_problem=matched,
        top_k=3,
    )

    assert result.candidates == ()
    assert result.paths
    assert all(path["pathSource"] == "inferred" for path in result.paths)
    assert all("storePath" not in path for path in result.paths)
    assert "not returned by Neo4j" in result.paths[0]["rationale"]


def test_exact_problem_matcher_recognizes_problem_id_source_id_and_title():
    matcher = ExactProblemMatcher((_uva_document(),))

    exact_id = matcher.match(QueryUnderstandingService().understand("UVA-10653"))
    exact_id_with_title = matcher.match(
        QueryUnderstandingService().understand("UVA-10653 - Bombs! NO they are Mines!!")
    )
    bare_source_id = matcher.match(QueryUnderstandingService().understand("10653"))
    exact_title = matcher.match(QueryUnderstandingService().understand("Bombs! NO they are Mines!!"))
    source_title_alias = matcher.match(QueryUnderstandingService().understand("UVa Bombs! NO they are Mines!!"))
    partial_title = matcher.match(QueryUnderstandingService().understand("Bombs mines shortest path"))

    assert exact_id is not None
    assert exact_id.problem_id == "uva-10653"
    assert exact_id.match_kind == "exact_problem_id"
    assert exact_id_with_title is not None
    assert exact_id_with_title.match_kind == "exact_problem_id"
    assert bare_source_id is not None
    assert bare_source_id.match_kind == "exact_source_id"
    assert exact_title is not None
    assert exact_title.match_kind == "exact_title"
    assert source_title_alias is not None
    assert source_title_alias.match_kind == "exact_title"
    assert partial_title is not None
    assert partial_title.match_kind == "partial_title"
    assert partial_title.confidence < exact_title.confidence


def test_exact_match_payload_seeds_statement_answer_and_hints_in_order():
    documents = (_uva_document(),)
    match = ExactProblemMatcher(documents).match(
        QueryUnderstandingService(documents).understand("UVA-10653 - Bombs! NO they are Mines!!")
    )

    assert match is not None
    payload = match.candidate.payload
    assert [
        chunk["payload"]["kind"]
        for chunk in payload["rawChunks"]
    ] == ["problem_card", "statement", "answer", "hint", "hint"]
    assert payload["chunkCount"] >= 3
    assert payload["rawChunksComplete"] is True
    assert payload["chunkEvidence"] == {
        "available": True,
        "complete": True,
        "missingSources": [],
        "unavailableReason": "",
    }


def test_partial_problem_match_does_not_override_python_input_kind():
    understanding = QueryUnderstandingService((_uva_document(),)).understand(
        "def solve():\n    bombs = mines = shortest = path = []"
    )

    assert understanding.input_kind == "python"


def test_online_pipeline_promotes_exact_problem_seed_without_polluting_similar_problems():
    documents = (
        _uva_document(),
        RetrievalDocument(
            id="leetcode-1091",
            source="LeetCode",
            source_id="1091",
            title="Shortest Path in Binary Matrix",
            text="Use BFS to find a shortest path in an unweighted binary matrix.",
            answer="Run BFS over eight directions.",
            concepts=("BFS", "Queue", "Visited Array"),
            problem_type="Graph Traversal",
        ),
    )

    result = OnlineQueryPipeline(
        documents=documents,
        vector_store=_FakeVectorStore(()),
        bm25_store=_FakeBM25Store(()),
    ).run("UVA-10653 - Bombs! NO they are Mines!!", top_k=2)
    evidence = EvidenceBuilder().build(
        result.reranked_candidates,
        result.graph_paths,
        matched_problem=result.matched_problem,
    )
    trace = result.trace.to_mapping()
    evidence_map = evidence.to_mapping()

    assert result.query_understanding.input_kind == "problem"
    assert result.matched_problem is not None
    assert result.matched_problem.problem_id == "uva-10653"
    assert trace["matchedProblem"]["id"] == "uva-10653"
    assert evidence_map["matchedProblem"]["id"] == "uva-10653"
    assert all(problem["id"] != "uva-10653" for problem in evidence_map["similarProblems"])
    assert all(candidate.id != "uva-10653" for candidate in result.graph_candidates)
    assert all(candidate.id != "uva-10653" for candidate in result.fused_candidates)
    assert all(candidate.id != "uva-10653" for candidate in result.reranked_candidates)
    assert _path_node_ids(result.graph_paths[0]) == [
        "uva-10653",
        "source:uva:10653",
        "concept:bfs",
    ]
    assert result.graph_paths[0]["pathSource"] == "inferred"


def test_online_pipeline_marks_exact_problem_graph_result_as_paths_only():
    document = RetrievalDocument(
        id="uva-1121",
        source="UVa",
        source_id="1121",
        title="Subsequence",
        text="Find the shortest contiguous subsequence with sum at least S.",
        answer="Use a sliding window.",
        concepts=("Sliding Window", "Two Pointers"),
        problem_type="Sliding Window",
        solution_hints=("Move right to expand the window.", "Move left while the sum stays large enough."),
    )

    result = OnlineQueryPipeline(documents=(document,)).run(
        "uva-1121 Subsequence",
        top_k=3,
    )
    trace = result.trace.to_mapping()

    assert result.matched_problem is not None
    assert result.graph_candidates == ()
    assert result.graph_paths
    assert trace["graphSearchStatus"] == "paths_only"


def test_matched_problem_scope_filters_similar_problems_without_reordering_remaining():
    matched = ExactProblemMatcher((_uva_document(),)).match(
        QueryUnderstandingService((_uva_document(),)).understand("UVA-10653 - Bombs! NO they are Mines!!")
    )
    assert matched is not None
    unrelated = RetrievalCandidate(
        id="leetcode-1091",
        title="Shortest Path in Binary Matrix",
        source="reranker",
        score=0.80,
        text="Use BFS with a queue over matrix states.",
        concepts=("BFS", "Queue"),
        problem_type="Graph Traversal",
    )
    weaker = RetrievalCandidate(
        id="leetcode-994",
        title="Rotting Oranges",
        source="reranker",
        score=0.60,
        text="Multi-source BFS with visited state tracking.",
        concepts=("BFS", "Queue", "State Tracking"),
        problem_type="Graph Traversal",
    )

    evidence = EvidenceBuilder().build(
        (unrelated, matched.candidate, weaker),
        (),
        matched_problem=matched,
        query_concepts=("BFS", "Shortest Path"),
    )
    evidence_map = evidence.to_mapping()

    assert evidence_map["matchedProblem"]["id"] == "uva-10653"
    assert [problem["id"] for problem in evidence_map["similarProblems"]] == [
        "leetcode-1091",
    ]
    assert "Queue" in evidence_map["dataStructureEvidence"]
    assert "Visited Array" not in evidence_map["dataStructureEvidence"]
    assert "Visited Array" in evidence_map["techniqueEvidence"]
    assert "State Tracking" not in evidence_map["techniqueEvidence"]
    assert evidence_map["commonMistakes"] == [
        "忘記標記 visited。",
        "queue 初始化錯誤，導致起點或距離沒有被正確設定。",
    ]


def test_online_pipeline_suppresses_partial_title_seed_for_python_input():
    result = OnlineQueryPipeline(documents=(_uva_document(),)).run(
        "def solve():\n    bombs = mines = shortest = path = []",
        top_k=1,
    )
    trace = result.trace.to_mapping()

    assert result.query_understanding.input_kind == "python"
    assert result.matched_problem is None
    assert trace["matchedProblem"] is None


def test_online_pipeline_suppresses_partial_title_seed_for_cpp_input():
    result = OnlineQueryPipeline(documents=(_uva_document(),)).run(
        """
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<int> bombs, mines, shortest, path;
    return 0;
}
""".strip(),
        top_k=1,
    )
    trace = result.trace.to_mapping()

    assert result.query_understanding.input_kind == "cpp"
    assert result.matched_problem is None
    assert trace["matchedProblem"] is None

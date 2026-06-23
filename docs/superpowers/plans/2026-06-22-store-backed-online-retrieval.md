# Store-Backed Online Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add store-backed online retrieval while keeping local document fallback, rule-based query understanding, and the current mock-only LLM path.

**Architecture:** `VectorSearchService`, `BM25SearchService`, and `GraphSearchService` accept optional stores and keep their current local-document behavior when stores are absent. `OnlineQueryPipeline` accepts `vector_store`, `bm25_store`, and `graph_store` injections and still builds `retrievalTrace`, `evidenceBundle`, and `contextPreview` from the same result objects.

**Tech Stack:** Python 3.11, pytest, FastAPI contracts, in-memory store adapters, `DeterministicMockEmbeddingProvider`, existing `RetrievalTrace` and `RetrievalEvidenceBundle` dataclasses.

---

## Scope Guardrails

- Do not add a real `LLMProvider`.
- Do not call an external LLM from retrieval, query understanding, reranking, evidence building, or context building.
- Leave `QueryUnderstandingService` rule-based.
- Leave `MockLLMProvider` and `LLMResponseGenerator` as the only response-generation path.
- Do not change the `/api/analysis` response shape except for store-backed retrieval internals feeding the existing fields.
- Do not implement FastAPI runtime store mode in this plan. Wiring actual Qdrant, Neo4j, or BM25Store injection into FastAPI belongs to the next phase: End-to-End Store-Backed Demo.

## Commit Policy

- Do not commit failing tests to `main`.
- Task 1 only verifies that the new tests fail because store injection is not supported yet.
- Do not make per-task commits during Tasks 3-6.
- Commit only after Tasks 3-6 are implemented and the full regression checks in Task 8 pass.
- The final commit in Task 8 is optional; if there are no new changes to commit, report `python -m pytest`, `.\scripts\quick-start.ps1 -Check`, and `git status --short` results instead.

## File Structure

- Modify: `backend/app/retrieval/pipeline.py`
  - Add optional `VectorStore`, `BM25Store`, and `GraphStore` usage to existing retrieval services.
  - Add store-hit-to-`RetrievalCandidate` helpers near `_candidate_from_document`.
  - Add store injection arguments to `OnlineQueryPipeline.__init__`.
- Modify: `tests/backend/test_online_retrieval_pipeline.py`
  - Add store-backed tests for vector, BM25, graph, and full pipeline injection.
  - Keep the current local document tests as fallback coverage.
- No runtime code changes are needed in `backend/app/main.py`.
- FastAPI runtime store mode is out of scope for this plan; `backend/app/main.py` should keep constructing `OnlineQueryPipeline()` without external store injection.
- No ingestion changes are needed unless a store-backed test exposes a missing payload field.

## Store Payload Contract Used By Retrieval

Store-backed retrieval should accept the payload shape already produced by ingestion chunks:

```python
{
    "problemId": "leetcode-994",
    "kind": "statement",
    "text": "Multi-source BFS with a queue on a grid.",
    "concepts": ["BFS", "Queue"],
    "metadata": {
        "source": "LeetCode",
        "sourceId": "994",
        "title": "Rotting Oranges",
        "problemType": "Graph Traversal",
    },
}
```

The candidate id used for fusion should be `problemId` when present, not the chunk id, so vector and BM25 hits for the same problem dedupe correctly.

---

### Task 1: Add Store-Backed Failing Tests

**Files:**
- Modify: `tests/backend/test_online_retrieval_pipeline.py`

- [ ] **Step 1: Extend imports for in-memory stores and store records**

Add these imports at the top of `tests/backend/test_online_retrieval_pipeline.py`:

```python
from backend.app.adapters.in_memory import InMemoryBM25Store, InMemoryGraphStore, InMemoryVectorStore
from backend.app.contracts import EntityRecord, RelationRecord
from backend.app.stores import BM25Document, VectorRecord
```

- [ ] **Step 2: Add store payload and indexing helpers after `_documents()`**

```python
def _store_payload(document: RetrievalDocument) -> dict[str, object]:
    return {
        "problemId": document.id,
        "kind": "statement",
        "text": document.text,
        "concepts": list(document.concepts),
        "metadata": {
            "source": document.source,
            "sourceId": document.source_id,
            "title": document.title,
            "problemType": document.problem_type,
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
```

- [ ] **Step 3: Add a vector store search test**

```python
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
```

- [ ] **Step 4: Add a BM25 store search test**

```python
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
```

- [ ] **Step 5: Add a graph store search test**

```python
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
    assert result.paths[0]["nodes"] == ["input", "concept:bfs", "leetcode-994"]
```

- [ ] **Step 6: Add a full pipeline store injection test**

```python
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
    assert trace["graphCandidates"]
    assert evidence.to_mapping()["similarProblems"]
    assert "Query Understanding" in context
```

- [ ] **Step 7: Run the new tests to confirm they fail for the expected reason**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py -v
```

Expected: FAIL with `TypeError` for unexpected `vector_store`, `bm25_store`, or `graph_store` keyword arguments.

- [ ] **Step 8: Do not commit the failing tests**

```powershell
git status --short
```

Expected: the new test changes remain uncommitted. Do not run `git add` or `git commit` while the tests are still failing.

---

### Task 2: Add Store Candidate Mapping Helpers

**Files:**
- Modify: `backend/app/retrieval/pipeline.py`

- [ ] **Step 1: Import store protocols and records**

Change the imports near the top of `backend/app/retrieval/pipeline.py` to include:

```python
from ..stores import BM25Store, GraphStore, SearchCandidate, VectorStore
```

- [ ] **Step 2: Add helper functions below `_candidate_from_document`**

```python
def _candidate_from_store_candidate(
    candidate: SearchCandidate,
    *,
    source: str,
) -> RetrievalCandidate:
    payload = dict(candidate.payload)
    metadata = _mapping(payload.get("metadata"))
    problem_id = str(
        payload.get("problemId")
        or payload.get("problem_id")
        or metadata.get("problemId")
        or candidate.id
    )
    concepts = _tuple_of_str(payload.get("concepts") or metadata.get("concepts"))
    title = str(payload.get("title") or metadata.get("title") or problem_id)
    problem_type = str(payload.get("problemType") or metadata.get("problemType") or "")
    text = str(payload.get("text") or payload.get("statement") or "")

    return RetrievalCandidate(
        id=problem_id,
        title=title,
        source=source,
        score=round(candidate.score, 6),
        text=text,
        concepts=concepts,
        problem_type=problem_type,
        payload={
            "storeCandidateId": candidate.id,
            "documentSource": str(payload.get("source") or metadata.get("source") or ""),
            "sourceId": str(payload.get("sourceId") or metadata.get("sourceId") or ""),
            "answer": str(payload.get("answer") or ""),
            "storePayload": payload,
        },
    )


def _mapping(value: Any) -> JsonMap:
    return dict(value) if isinstance(value, dict) else {}


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)
```

- [ ] **Step 3: Run the targeted tests**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_vector_search_service_can_use_vector_store -v
```

Expected: still FAIL because `VectorSearchService.__init__` does not accept `vector_store` yet.

---

### Task 3: Add `VectorSearchService` VectorStore Support

**Files:**
- Modify: `backend/app/retrieval/pipeline.py`

- [ ] **Step 1: Replace `VectorSearchService.__init__` and `search`**

```python
class VectorSearchService:
    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._documents = tuple(documents)
        self._embedding_provider = embedding_provider or DeterministicMockEmbeddingProvider()
        self._vector_store = vector_store
        self._vectors = {
            document.id: self._embedding_provider.embed_text(
                f"{document.title} {document.text} {' '.join(document.concepts)}"
            )
            for document in self._documents
        }

    def search(self, understanding: QueryUnderstanding, *, top_k: int) -> tuple[RetrievalCandidate, ...]:
        query_vector = self._embedding_provider.embed_text(understanding.normalized_query)
        if self._vector_store is not None:
            return tuple(
                _candidate_from_store_candidate(candidate, source="vector")
                for candidate in self._vector_store.search(query_vector, top_k=top_k)
            )

        candidates = [
            _candidate_from_document(
                document,
                source="vector",
                score=_cosine(query_vector, self._vectors[document.id]),
            )
            for document in self._documents
        ]
        return tuple(sorted(candidates, key=lambda item: (-item.score, item.id))[:top_k])
```

- [ ] **Step 2: Run the vector store test**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_vector_search_service_can_use_vector_store -v
```

Expected: PASS.

- [ ] **Step 3: Run the existing local vector/BM25/graph test**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_vector_graph_and_bm25_search_return_candidates -v
```

Expected: PASS, proving local document fallback still works.

- [ ] **Step 4: Checkpoint without committing**

```powershell
git status --short
```

Expected: vector store code and tests may remain modified locally. Do not commit until Tasks 3-6 are done and Task 8 passes.

---

### Task 4: Add `BM25SearchService` BM25Store Support

**Files:**
- Modify: `backend/app/retrieval/pipeline.py`

- [ ] **Step 1: Replace `BM25SearchService`**

```python
class BM25SearchService:
    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        bm25_store: BM25Store | None = None,
    ) -> None:
        self._documents = tuple(documents)
        self._bm25_store = bm25_store

    def search(self, understanding: QueryUnderstanding, *, top_k: int) -> tuple[RetrievalCandidate, ...]:
        if self._bm25_store is not None:
            return tuple(
                _candidate_from_store_candidate(candidate, source="bm25")
                for candidate in self._bm25_store.search(understanding.normalized_query, top_k=top_k)
            )

        query_terms = set(understanding.keywords)
        candidates: list[RetrievalCandidate] = []
        for document in self._documents:
            terms = _tokens(
                f"{document.title} {document.text} {document.answer} {' '.join(document.concepts)}"
            )
            if not query_terms:
                score = 0.0
            else:
                score = sum(1 for term in terms if term in query_terms) / max(len(terms), 1)
                score += len(query_terms & set(terms)) / len(query_terms)
            candidates.append(_candidate_from_document(document, source="bm25", score=score))
        return tuple(sorted(candidates, key=lambda item: (-item.score, item.id))[:top_k])
```

- [ ] **Step 2: Run the BM25 store test**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_bm25_search_service_can_use_bm25_store -v
```

Expected: PASS.

- [ ] **Step 3: Run the existing local fallback test**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_vector_graph_and_bm25_search_return_candidates -v
```

Expected: PASS.

- [ ] **Step 4: Checkpoint without committing**

```powershell
git status --short
```

Expected: BM25 store code and tests may remain modified locally. Do not commit until Tasks 3-6 are done and Task 8 passes.

---

### Task 5: Add `GraphSearchService` GraphStore Support

**Files:**
- Modify: `backend/app/retrieval/pipeline.py`

- [ ] **Step 1: Replace `GraphSearchService.__init__` and branch `search`**

```python
class GraphSearchService:
    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        graph_store: GraphStore | None = None,
    ) -> None:
        self._documents = tuple(documents)
        self._graph_store = graph_store

    def search(self, linked_entities: Sequence[JsonMap], *, top_k: int) -> GraphSearchResult:
        if self._graph_store is not None:
            return self._search_store(linked_entities, top_k=top_k)

        entity_names = {str(entity["name"]).lower() for entity in linked_entities}
        entity_ids = {str(entity["entityId"]) for entity in linked_entities}
        candidates: list[RetrievalCandidate] = []
        paths: list[JsonMap] = []
        for document in self._documents:
            matched: list[JsonMap] = []
            for concept in document.concepts:
                concept_id = f"concept:{_slug(concept)}"
                if concept.lower() in entity_names or concept_id in entity_ids:
                    matched.append({"entityId": concept_id, "name": concept})
            pattern_id = f"pattern:{_slug(document.problem_type)}"
            if document.problem_type.lower() in entity_names or pattern_id in entity_ids:
                matched.append({"entityId": pattern_id, "name": document.problem_type})
            if not matched:
                continue
            score = len(matched) / max(len(linked_entities), 1)
            candidates.append(_candidate_from_document(document, source="graph", score=score))
            for match in matched:
                paths.append(
                    {
                        "nodes": ["input", match["entityId"], document.id],
                        "relations": ["MENTIONS", "REQUIRED_BY"],
                        "score": round(score, 6),
                        "rationale": f"linked {match['name']} to {document.title}",
                    }
                )
        return GraphSearchResult(
            candidates=tuple(sorted(candidates, key=lambda item: (-item.score, item.id))[:top_k]),
            paths=tuple(paths),
        )
```

- [ ] **Step 2: Add `_search_store` inside `GraphSearchService`**

```python
    def _search_store(self, linked_entities: Sequence[JsonMap], *, top_k: int) -> GraphSearchResult:
        assert self._graph_store is not None
        candidates: list[RetrievalCandidate] = []
        paths: list[JsonMap] = []
        for document in self._documents:
            document_paths: list[JsonMap] = []
            for entity in linked_entities:
                entity_id = str(entity["entityId"])
                direct_paths = self._graph_store.find_paths(document.id, entity_id, max_hops=3)
                reverse_paths = self._graph_store.find_paths(entity_id, document.id, max_hops=3)
                for path in (*direct_paths, *reverse_paths):
                    normalized = _normalize_graph_store_path(
                        path,
                        entity=entity,
                        document=document,
                    )
                    document_paths.append(normalized)
                    paths.append(normalized)

            if not document_paths:
                continue
            score = max(float(path.get("score", 0.0)) for path in document_paths)
            candidates.append(_candidate_from_document(document, source="graph", score=score))

        return GraphSearchResult(
            candidates=tuple(sorted(candidates, key=lambda item: (-item.score, item.id))[:top_k]),
            paths=tuple(paths),
        )
```

- [ ] **Step 3: Add graph path normalization helper near the other helpers**

```python
def _normalize_graph_store_path(
    path: JsonMap,
    *,
    entity: JsonMap,
    document: RetrievalDocument,
) -> JsonMap:
    entity_id = str(entity["entityId"])
    entity_name = str(entity.get("name", entity_id))
    raw_nodes = [str(node) for node in path.get("nodes", [])]
    raw_relations = [str(relation) for relation in path.get("relations", [])]

    return {
        "nodes": ["input", entity_id, document.id],
        "relations": ["MENTIONS", "REQUIRED_BY"],
        "score": round(float(path.get("score", 0.0)), 6),
        "rationale": f"linked {entity_name} to {document.title}",
        "storePath": {
            "nodes": raw_nodes,
            "relations": raw_relations,
        },
    }
```

`nodes` and `relations` are the stable summary path used by the API/debug contract: `input -> linked entity -> problem`. `storePath.nodes` and `storePath.relations` preserve the original graph store path for inspection.

- [ ] **Step 4: Run the graph store test**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_graph_search_service_can_use_graph_store -v
```

Expected: PASS.

- [ ] **Step 5: Run all online retrieval tests**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py -v
```

Expected: all service-level tests PASS; `test_online_pipeline_accepts_store_injection_and_preserves_debug_outputs` still FAILS with an unexpected store-injection keyword until Task 6 is implemented.

- [ ] **Step 6: Checkpoint without committing**

```powershell
git status --short
```

Expected: graph store code and tests may remain modified locally. Do not commit until Tasks 3-6 are done and Task 8 passes.

---

### Task 6: Add Store Injection To `OnlineQueryPipeline`

**Files:**
- Modify: `backend/app/retrieval/pipeline.py`

- [ ] **Step 1: Replace `OnlineQueryPipeline.__init__`**

```python
class OnlineQueryPipeline:
    def __init__(
        self,
        *,
        documents: Sequence[RetrievalDocument] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        self._documents = tuple(documents) if documents is not None else _load_default_documents()
        self._embedding_provider = embedding_provider or DeterministicMockEmbeddingProvider()
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._graph_store = graph_store
```

- [ ] **Step 2: Update service construction inside `OnlineQueryPipeline.run`**

Use this block in `run` where the three retrieval services are constructed:

```python
        vector_candidates = VectorSearchService(
            self._documents,
            self._embedding_provider,
            vector_store=self._vector_store,
        ).search(understanding, top_k=max(top_k * 2, top_k))
        bm25_candidates = BM25SearchService(
            self._documents,
            bm25_store=self._bm25_store,
        ).search(
            understanding,
            top_k=max(top_k * 2, top_k),
        )
        graph_result = GraphSearchService(
            self._documents,
            graph_store=self._graph_store,
        ).search(
            linked_entities,
            top_k=max(top_k * 2, top_k),
        )
```

- [ ] **Step 3: Run the full pipeline injection test**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_online_pipeline_accepts_store_injection_and_preserves_debug_outputs -v
```

Expected: PASS.

- [ ] **Step 4: Run all online retrieval pipeline tests**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py -v
```

Expected: PASS.

- [ ] **Step 5: Checkpoint without committing**

```powershell
git status --short
```

Expected: pipeline injection code and tests may remain modified locally. Do not commit until the full regression checks in Task 8 pass.

---

### Task 7: Confirm LLM And Query Understanding Scope Did Not Change

**Files:**
- Read: `backend/app/retrieval/pipeline.py`
- Read: `backend/app/services/llm.py`
- Test: `tests/backend/test_llm.py`
- Test: `tests/backend/test_online_retrieval_pipeline.py`

- [ ] **Step 1: Inspect LLM files for accidental provider expansion**

Run:

```powershell
rg -n "LLMProvider|MockLLMProvider|LLMResponseGenerator|OpenAI|Anthropic|api_key|apiKey" backend tests
```

Expected:
- `MockLLMProvider` remains in `backend/app/services/llm.py`.
- `LLMResponseGenerator` remains in `backend/app/retrieval/pipeline.py`.
- No real provider, API key, OpenAI, Anthropic, or network client is introduced.

- [ ] **Step 2: Inspect Query Understanding for rule-based behavior**

Run:

```powershell
rg -n "class QueryUnderstandingService|def understand|detect_input_kind|LLM" backend/app/retrieval/pipeline.py backend/app/analysis.py
```

Expected:
- `QueryUnderstandingService.understand()` still uses string normalization, `_tokens`, and `detect_input_kind`.
- `QueryUnderstandingService` does not accept or call any LLM object.

- [ ] **Step 3: Run LLM and online retrieval tests**

Run:

```powershell
python -m pytest tests/backend/test_llm.py tests/backend/test_online_retrieval_pipeline.py -v
```

Expected: PASS.

---

### Task 8: Full Regression Verification

**Files:**
- Test: all backend tests under `tests/`

- [ ] **Step 1: Run the full pytest suite**

Run:

```powershell
python -m pytest
```

Expected: PASS.

- [ ] **Step 2: Run the fast repo preflight if no pytest failure remains**

Run:

```powershell
.\scripts\quick-start.ps1 -Check
```

Expected: PASS. If this script reports frontend checks too, preserve the reported command names in the final execution summary.

- [ ] **Step 3: Check git diff**

Run:

```powershell
git diff -- backend/app/retrieval/pipeline.py tests/backend/test_online_retrieval_pipeline.py
git status --short
```

Expected:
- Runtime changes are limited to `backend/app/retrieval/pipeline.py`.
- Test changes are limited to `tests/backend/test_online_retrieval_pipeline.py`.
- No frontend, API schema, or LLM provider files changed.

- [ ] **Step 4: Optional final commit**

```powershell
git status --short
```

Expected: if `git status --short` shows no changes, do not commit. Report the `python -m pytest`, `.\scripts\quick-start.ps1 -Check`, and `git status --short` results.

If implementation changes are present and the user wants a commit, run:

```powershell
git add backend/app/retrieval/pipeline.py tests/backend/test_online_retrieval_pipeline.py
git commit -m "feat: support store-backed online retrieval"
```

---

## Self-Review

- Spec coverage:
  - `VectorSearchService` supports `VectorStore`: Task 3.
  - `BM25SearchService` supports `BM25Store`: Task 4.
  - `GraphSearchService` supports `GraphStore`: Task 5.
  - `OnlineQueryPipeline` supports `vector_store`, `bm25_store`, and `graph_store` injection: Task 6.
  - Local documents fallback stays active: Tasks 3, 4, 5, and existing local tests.
  - Store-backed tests are added: Task 1.
  - `python -m pytest` stays green: Task 8.
  - Real LLM is excluded: Task 7.
  - Query Understanding stays rule-based: Task 7.
  - FastAPI runtime store mode is excluded and deferred to End-to-End Store-Backed Demo: Scope Guardrails.
  - Graph store debug paths preserve `storePath.nodes` and `storePath.relations` while keeping stable summary nodes and relations: Task 5.
- Placeholder scan:
  - No placeholder tasks are intentionally left open.
  - Every code-changing task includes the concrete code block to apply.
- Type consistency:
  - Store protocols come from `backend.app.stores`.
  - Store search hits are converted through `SearchCandidate`.
  - Existing `RetrievalCandidate`, `GraphSearchResult`, `OnlineQueryResult`, and `RetrievalTrace` remain the public internal result types.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-22-store-backed-online-retrieval.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using `superpowers:executing-plans`, with checkpoints after each task group.

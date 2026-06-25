# Exact Problem Query Retrieval Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make exact problem queries such as `UVA-10653 - Bombs! NO they are Mines!!` resolve to a dedicated Matched Problem, while keeping vector/BM25/graph/hybrid debug evidence honest and inspectable.

**Architecture:** Add an exact-problem retrieval seed before vector, graph, BM25, fusion, and evidence building. Keep Matched Problem separate from Similar Problems, aggregate chunk-level store hits by problem while preserving `rawChunks`, and make graph paths explicitly distinguish real Neo4j paths from inferred fallback evidence.

**Tech Stack:** FastAPI/Pydantic backend, Python retrieval pipeline, processed JSON ingestion artifacts, Qdrant/Neo4j/BM25 adapters, React/TypeScript frontend, pytest, ruff, Vite build.

---

## Progressive Delivery Shape

This plan is intentionally split so each phase can be verified independently:

1. Backend contract and exact matching: proves `UVA-10653`, `10653`, exact title, and partial title are classified correctly.
2. Chunk aggregation: removes repeated problem rows while preserving raw chunk debug data.
3. Graph honesty: makes real Neo4j paths and inferred paths distinguishable and consistent across Graph Evidence and Graph Trace.
4. Fusion/evidence separation: pins only Matched Problem and keeps Similar Problems ranking unpolluted.
5. Runtime artifact/provider metadata: fixes BM25 alias indexing and marks mock versus real providers.
6. Frontend contract/rendering: shows Matched Problem independently in all retrieval modes, displays `rawChunks`, fixes duplicate keys, and localizes visible debug labels to Traditional Chinese.
7. Integration verification: proves the original `UVA-10653 - Bombs! NO they are Mines!!` demo behavior through API and browser.

The AGENTS.md Git rule overrides the writing-plans default commit cadence. Do not commit during execution unless the user explicitly asks for commits. Use `git diff --check` and `git status --short --branch` as review checkpoints instead.

## File Map

- Modify: `backend/app/contracts.py`
  - Add `matched_problem` to `RetrievalTrace` and `RetrievalEvidenceBundle`.
  - Add optional `technique_evidence` or `state_tracking_evidence` so `Visited Array` can leave `dataStructureEvidence`.
- Modify: `backend/app/retrieval/pipeline.py`
  - Add `ExactProblemMatch` and `ExactProblemMatcher`.
  - Thread exact match through `OnlineQueryPipeline.run()`.
  - Aggregate vector and BM25 candidates by problem id with `rawChunks`.
  - Seed graph search with exact problem node.
  - Keep Matched Problem out of Similar Problems.
  - Add `pathSource` to graph paths.
  - Localize context builder labels.
- Modify: `backend/app/retrieval/runtime.py`
  - Extend runtime source/provider metadata and debug trace labeling.
- Modify: `backend/app/providers.py`
  - Expose deterministic mock provider kind without changing the existing `model_name`.
- Modify: `backend/app/ingestion/pipeline.py`
  - Enrich BM25 searchable text with problem id, source id, source alias, title, and concepts.
  - Keep title matching semantics in `ExactProblemMatcher`, not in BM25.
  - Reclassify `Visited Array` as state tracking.
- Modify: `backend/app/main.py`
  - Expose `matchedProblem` in `AnalysisResponse`.
  - Build top-level Graph Evidence from retrieval graph paths when available.
  - Preserve debug-only runtime metadata.
- Modify: `frontend/src/types.ts`
  - Add `MatchedProblem`, `ProviderDescriptor`, `pathSource`, and `rawChunks` types.
- Modify: `frontend/src/api.ts`
  - Normalize the new fields, preserve raw chunk payloads, and update fallback mock data.
- Modify: `frontend/src/App.tsx`
  - Add a Matched Problem panel independent of retrieval mode.
  - Fix candidate keys with store candidate id/raw chunk ids.
  - Localize visible debug labels.
  - Render inferred graph paths clearly.
- Modify tests:
  - `tests/backend/test_contracts_and_providers.py`
  - `tests/backend/test_online_retrieval_pipeline.py`
  - `tests/backend/test_runtime_retrieval.py`
  - `tests/backend/test_analysis.py`
  - `tests/backend/test_processed_problem_loader.py`
  - Add `frontend` build verification with `npm.cmd run build`; add frontend unit tests only if the repo already has a test runner configured.

---

### Task 1: Exact Problem Contract And Matcher

**Files:**
- Modify: `backend/app/retrieval/pipeline.py:35-108`
- Modify: `backend/app/contracts.py:173-212`
- Modify: `tests/backend/test_online_retrieval_pipeline.py`
- Modify: `tests/backend/test_contracts_and_providers.py`

- [ ] **Step 1: Add failing exact matcher tests**

Append these tests to `tests/backend/test_online_retrieval_pipeline.py`:

```python
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


def test_exact_problem_matcher_recognizes_problem_id_source_id_and_title():
    matcher = ExactProblemMatcher((_uva_document(),))

    exact_id = matcher.match(QueryUnderstandingService().understand("UVA-10653 - Bombs! NO they are Mines!!"))
    bare_source_id = matcher.match(QueryUnderstandingService().understand("10653"))
    exact_title = matcher.match(QueryUnderstandingService().understand("Bombs! NO they are Mines!!"))
    partial_title = matcher.match(QueryUnderstandingService().understand("Bombs mines shortest path"))

    assert exact_id is not None
    assert exact_id.problem_id == "uva-10653"
    assert exact_id.match_kind == "exact_problem_id"
    assert bare_source_id is not None
    assert bare_source_id.match_kind == "exact_source_id"
    assert exact_title is not None
    assert exact_title.match_kind == "exact_title"
    assert partial_title is not None
    assert partial_title.match_kind == "partial_title"
    assert partial_title.confidence < exact_title.confidence
```

Update the import block in `tests/backend/test_online_retrieval_pipeline.py`:

```python
from backend.app.retrieval.pipeline import (
    BM25SearchService,
    ContextBuilder,
    EntityLinkingService,
    EvidenceBuilder,
    ExactProblemMatcher,
    GraphSearchService,
    HybridFusionService,
    OnlineQueryPipeline,
    QueryUnderstandingService,
    Reranker,
    RetrievalCandidate,
    RetrievalDocument,
    VectorSearchService,
)
```

- [ ] **Step 2: Run the targeted failing test**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_exact_problem_matcher_recognizes_problem_id_source_id_and_title -q
```

Expected: FAIL with `ImportError` or `NameError` for `ExactProblemMatcher`.

- [ ] **Step 3: Add exact match dataclass and matcher**

In `backend/app/retrieval/pipeline.py`, add `Literal` to the imports:

```python
from typing import Any, Literal, Sequence
```

Add these dataclasses after `QueryUnderstanding`:

```python
ExactMatchKind = Literal["exact_problem_id", "exact_source_id", "exact_title", "partial_title"]


@dataclass(frozen=True)
class ExactProblemMatch:
    problem_id: str
    title: str
    source: str
    source_id: str
    match_kind: ExactMatchKind
    confidence: float
    candidate: RetrievalCandidate

    def to_mapping(self) -> JsonMap:
        return {
            "id": self.problem_id,
            "title": self.title,
            "source": self.source,
            "sourceId": self.source_id,
            "matchKind": self.match_kind,
            "confidence": round(self.confidence, 6),
            "score": round(self.candidate.score, 6),
            "sharedConcepts": list(self.candidate.concepts),
            "problemType": self.candidate.problem_type,
            "answerHint": self.candidate.payload.get("answer", ""),
            "solutionHints": list(_tuple_of_str(self.candidate.payload.get("solutionHints"))),
            "difficulty": str(self.candidate.payload.get("difficulty") or ""),
            "constraints": list(_tuple_of_str(self.candidate.payload.get("constraints"))),
        }
```

Move `RetrievalCandidate` above `ExactProblemMatch` if Python name resolution requires it. Keep the public mapping keys camelCase because the API and frontend already use camelCase.

Add this matcher after `QueryUnderstandingService`:

```python
class ExactProblemMatcher:
    def __init__(self, documents: Sequence[RetrievalDocument]) -> None:
        self._documents = tuple(documents)

    def match(self, understanding: QueryUnderstanding) -> ExactProblemMatch | None:
        query = _normalize_alias(understanding.normalized_query)
        if not query:
            return None

        best: ExactProblemMatch | None = None
        for document in self._documents:
            candidate = _candidate_from_document(document, source="exact", score=1.0)
            aliases = _problem_aliases(document)
            exact_problem_ids = {_normalize_alias(document.id), _normalize_alias(document.id.replace("-", " "))}
            exact_source_ids = {
                _normalize_alias(document.source_id),
                _normalize_alias(f"{document.source}-{document.source_id}"),
                _normalize_alias(f"{document.source} {document.source_id}"),
            }
            exact_title = _normalize_alias(document.title)
            title_tokens = set(_tokens(document.title))
            query_tokens = set(understanding.keywords)

            if query in exact_problem_ids:
                match = ExactProblemMatch(document.id, document.title, document.source, document.source_id, "exact_problem_id", 1.0, candidate)
            elif query in exact_source_ids:
                match = ExactProblemMatch(document.id, document.title, document.source, document.source_id, "exact_source_id", 0.98, candidate)
            elif query == exact_title or query in aliases:
                match = ExactProblemMatch(document.id, document.title, document.source, document.source_id, "exact_title", 0.96, candidate)
            elif title_tokens and len(title_tokens & query_tokens) >= min(2, len(title_tokens)):
                confidence = len(title_tokens & query_tokens) / len(title_tokens)
                match = ExactProblemMatch(document.id, document.title, document.source, document.source_id, "partial_title", round(0.70 + (0.20 * confidence), 6), candidate)
            else:
                continue

            if best is None or match.confidence > best.confidence:
                best = match
        return best
```

Add helpers near `_tokens()`:

```python
def _normalize_alias(value: str) -> str:
    return " ".join(_tokens(value))


def _problem_aliases(document: RetrievalDocument) -> set[str]:
    return {
        _normalize_alias(document.title),
        _normalize_alias(f"{document.source} {document.source_id} {document.title}"),
        _normalize_alias(f"{document.source}-{document.source_id} {document.title}"),
        _normalize_alias(f"{document.id} {document.title}"),
    }
```

- [ ] **Step 4: Make query understanding use exact match for `inputKind`**

Change `QueryUnderstandingService.understand()` to accept documents:

```python
class QueryUnderstandingService:
    def __init__(self, documents: Sequence[RetrievalDocument] = ()) -> None:
        self._documents = tuple(documents)

    def understand(self, query: str) -> QueryUnderstanding:
        normalized = " ".join(query.strip().split())
        lowered = normalized.lower()
        keywords = tuple(dict.fromkeys(_tokens(lowered)))
        input_kind = detect_input_kind(query)
        if ExactProblemMatcher(self._documents).match(
            QueryUnderstanding(
                original_query=query,
                normalized_query=normalized,
                input_kind=input_kind,
                intent="problem_search",
                keywords=keywords,
            )
        ):
            input_kind = "problem"
        intent = "code_analysis" if input_kind in {"cpp", "python"} else "problem_search"
        return QueryUnderstanding(
            original_query=query,
            normalized_query=normalized,
            input_kind=input_kind,
            intent=intent,
            keywords=keywords,
        )
```

This keeps `inputKind` within the existing enum and puts the exact match detail in `matchedProblem.matchKind`.

- [ ] **Step 5: Extend trace and evidence contracts for Matched Problem**

In `backend/app/contracts.py`, update `RetrievalTrace`:

```python
@dataclass(frozen=True)
class RetrievalTrace:
    query_understanding: JsonMap = field(default_factory=dict)
    entity_linking: list[JsonMap] = field(default_factory=list)
    matched_problem: JsonMap | None = None
    vector_candidates: list[JsonMap] = field(default_factory=list)
    graph_candidates: list[JsonMap] = field(default_factory=list)
    bm25_candidates: list[JsonMap] = field(default_factory=list)
    fusion_scores: list[JsonMap] = field(default_factory=list)
    reranker_scores: list[JsonMap] = field(default_factory=list)

    def to_mapping(self) -> JsonMap:
        return {
            "queryUnderstanding": dict(self.query_understanding),
            "entityLinking": [dict(item) for item in self.entity_linking],
            "matchedProblem": dict(self.matched_problem) if self.matched_problem else None,
            "vectorCandidates": [dict(item) for item in self.vector_candidates],
            "graphCandidates": [dict(item) for item in self.graph_candidates],
            "bm25Candidates": [dict(item) for item in self.bm25_candidates],
            "fusionScores": [dict(item) for item in self.fusion_scores],
            "rerankerScores": [dict(item) for item in self.reranker_scores],
        }
```

Update `RetrievalEvidenceBundle`:

```python
@dataclass(frozen=True)
class RetrievalEvidenceBundle:
    matched_problem: JsonMap | None = None
    similar_problems: list[JsonMap] = field(default_factory=list)
    graph_paths: list[JsonMap] = field(default_factory=list)
    algorithm_evidence: list[str] = field(default_factory=list)
    data_structure_evidence: list[str] = field(default_factory=list)
    technique_evidence: list[str] = field(default_factory=list)
    pattern_evidence: list[str] = field(default_factory=list)
    common_mistakes: list[str] = field(default_factory=list)

    def to_mapping(self) -> JsonMap:
        return {
            "matchedProblem": dict(self.matched_problem) if self.matched_problem else None,
            "similarProblems": [dict(item) for item in self.similar_problems],
            "graphPaths": [dict(item) for item in self.graph_paths],
            "algorithmEvidence": list(self.algorithm_evidence),
            "dataStructureEvidence": list(self.data_structure_evidence),
            "techniqueEvidence": list(self.technique_evidence),
            "patternEvidence": list(self.pattern_evidence),
            "commonMistakes": list(self.common_mistakes),
        }
```

Update `tests/backend/test_contracts_and_providers.py::test_retrieval_trace_and_evidence_bundle_have_expected_debug_shape` so the key set includes `matchedProblem` and evidence mapping contains `matchedProblem`.

- [ ] **Step 6: Run the exact matcher and contract tests**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_exact_problem_matcher_recognizes_problem_id_source_id_and_title tests/backend/test_contracts_and_providers.py::test_retrieval_trace_and_evidence_bundle_have_expected_debug_shape -q
```

Expected: PASS.

### Task 2: Exact Seed Through Online Pipeline

**Files:**
- Modify: `backend/app/retrieval/pipeline.py:83-92`
- Modify: `backend/app/retrieval/pipeline.py:460-524`
- Modify: `tests/backend/test_online_retrieval_pipeline.py`

- [ ] **Step 1: Add failing end-to-end pipeline test for exact query**

Append to `tests/backend/test_online_retrieval_pipeline.py`:

```python
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

    result = OnlineQueryPipeline(documents=documents).run("UVA-10653 - Bombs! NO they are Mines!!", top_k=2)
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
```

- [ ] **Step 2: Run the failing pipeline test**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_online_pipeline_promotes_exact_problem_seed_without_polluting_similar_problems -q
```

Expected: FAIL because `OnlineQueryResult` has no `matched_problem` and `EvidenceBuilder.build()` has no `matched_problem` argument.

- [ ] **Step 3: Add `matched_problem` to pipeline result**

Update `OnlineQueryResult` in `backend/app/retrieval/pipeline.py`:

```python
@dataclass(frozen=True)
class OnlineQueryResult:
    query_understanding: QueryUnderstanding
    linked_entities: tuple[JsonMap, ...]
    matched_problem: ExactProblemMatch | None
    vector_candidates: tuple[RetrievalCandidate, ...]
    graph_candidates: tuple[RetrievalCandidate, ...]
    bm25_candidates: tuple[RetrievalCandidate, ...]
    fused_candidates: tuple[RetrievalCandidate, ...]
    reranked_candidates: tuple[RetrievalCandidate, ...]
    graph_paths: tuple[JsonMap, ...]
    trace: RetrievalTrace
```

- [ ] **Step 4: Thread exact seed through `OnlineQueryPipeline.run()`**

Replace the first lines of `run()` with this shape:

```python
    def run(self, query: str, *, top_k: int = 5) -> OnlineQueryResult:
        understanding = QueryUnderstandingService(self._documents).understand(query)
        matched_problem = ExactProblemMatcher(self._documents).match(understanding)
        linked_entities = EntityLinkingService().link(understanding, matched_problem=matched_problem)
```

When constructing `RetrievalTrace`, add:

```python
            matched_problem=matched_problem.to_mapping() if matched_problem else None,
```

When returning `OnlineQueryResult`, add:

```python
            matched_problem=matched_problem,
```

- [ ] **Step 5: Extend entity linking with problem seed**

Change `EntityLinkingService.link()` signature and append the exact problem entity:

```python
    def link(
        self,
        understanding: QueryUnderstanding,
        *,
        matched_problem: ExactProblemMatch | None = None,
    ) -> tuple[JsonMap, ...]:
        text = understanding.normalized_query.lower()
        linked: list[JsonMap] = []
        if matched_problem is not None:
            linked.append(
                {
                    "entityId": matched_problem.problem_id,
                    "name": matched_problem.title,
                    "type": "problem",
                    "confidence": matched_problem.confidence,
                    "matchKind": matched_problem.match_kind,
                }
            )
        for entity_id, name, kind in self._concept_aliases:
            terms = _tokens(name)
            if any(term in understanding.keywords for term in terms) or name.lower() in text:
                linked.append(
                    {
                        "entityId": entity_id,
                        "name": name,
                        "type": kind,
                        "confidence": 1.0,
                    }
                )
        return tuple(linked)
```

- [ ] **Step 6: Update `EvidenceBuilder.build()` to accept Matched Problem**

Change the signature:

```python
    def build(
        self,
        candidates: Sequence[RetrievalCandidate],
        graph_paths: Sequence[JsonMap],
        *,
        matched_problem: ExactProblemMatch | None = None,
    ) -> RetrievalEvidenceBundle:
```

Filter the matched id out of similar candidates:

```python
        matched_problem_id = matched_problem.problem_id if matched_problem else ""
        similar_candidates = [candidate for candidate in candidates if candidate.id != matched_problem_id]
```

Use `similar_candidates` in the `similar_problems=[...]` comprehension and pass:

```python
            matched_problem=matched_problem.to_mapping() if matched_problem else None,
```

- [ ] **Step 7: Update main API call site**

In `backend/app/main.py`, change:

```python
    evidence_bundle = EvidenceBuilder().build(
        pipeline_result.reranked_candidates,
        pipeline_result.graph_paths,
    )
```

to:

```python
    evidence_bundle = EvidenceBuilder().build(
        pipeline_result.reranked_candidates,
        pipeline_result.graph_paths,
        matched_problem=pipeline_result.matched_problem,
    )
```

- [ ] **Step 8: Run affected backend tests**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py tests/backend/test_analysis.py -q
```

Expected: PASS after updating existing tests that call `EvidenceBuilder().build(...)` to pass `matched_problem=` only when they assert exact-match behavior.

### Task 3: Aggregate Chunk Candidates But Preserve `rawChunks`

**Files:**
- Modify: `backend/app/retrieval/pipeline.py:138-202`
- Modify: `backend/app/retrieval/pipeline.py:549-584`
- Modify: `tests/backend/test_online_retrieval_pipeline.py`

- [ ] **Step 1: Add failing aggregation test**

Append to `tests/backend/test_online_retrieval_pipeline.py`:

```python
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
```

Update imports for the private helper only in tests if the project accepts testing private helpers:

```python
from backend.app.retrieval.pipeline import _aggregate_problem_candidates
```

If private helper import is not desired, test through fake store search results instead. Keep the same assertions.

- [ ] **Step 2: Run the failing aggregation test**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_store_chunk_candidates_are_aggregated_by_problem_and_keep_raw_chunks -q
```

Expected: FAIL because `_aggregate_problem_candidates` does not exist.

- [ ] **Step 3: Add aggregation helper**

Add this helper near `_candidate_from_store_candidate()`:

```python
def _aggregate_problem_candidates(
    candidates: Sequence[RetrievalCandidate],
    *,
    source: str,
    top_k: int,
) -> tuple[RetrievalCandidate, ...]:
    grouped: dict[str, list[RetrievalCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.id, []).append(candidate)

    aggregated: list[RetrievalCandidate] = []
    for problem_id, problem_candidates in grouped.items():
        ordered_chunks = sorted(problem_candidates, key=lambda item: (-item.score, str(item.payload.get("storeCandidateId", ""))))
        best = ordered_chunks[0]
        payload = dict(best.payload)
        payload["rawChunks"] = [chunk.to_mapping() for chunk in ordered_chunks]
        payload["chunkCount"] = len(ordered_chunks)
        payload["sources"] = [source]
        aggregated.append(replace(best, score=round(best.score, 6), payload=payload))

    return tuple(sorted(aggregated, key=lambda item: (-item.score, item.id))[:top_k])
```

- [ ] **Step 4: Use aggregation in store-backed vector and BM25 paths**

In `VectorSearchService.search()`, replace the store branch with:

```python
        if self._vector_store is not None:
            raw_candidates = tuple(
                _candidate_from_store_candidate(candidate, source="vector")
                for candidate in self._vector_store.search(query_vector, top_k=top_k)
            )
            return _aggregate_problem_candidates(raw_candidates, source="vector", top_k=top_k)
```

In `BM25SearchService.search()`, replace the store branch with:

```python
        if self._bm25_store is not None:
            raw_candidates = tuple(
                _candidate_from_store_candidate(candidate, source="bm25")
                for candidate in self._bm25_store.search(understanding.normalized_query, top_k=top_k)
                if candidate.score > 0
            )
            return _aggregate_problem_candidates(raw_candidates, source="bm25", top_k=top_k)
```

The `candidate.score > 0` filter prevents zero-score BM25 rows from being shown as meaningful lexical matches.

- [ ] **Step 5: Update React duplicate-key expectation through backend trace**

Add this assertion to the store injection test in `tests/backend/test_online_retrieval_pipeline.py` after trace construction:

```python
    assert "rawChunks" in trace["vectorCandidates"][0]["payload"]
    assert "rawChunks" in trace["bm25Candidates"][0]["payload"]
```

- [ ] **Step 6: Run focused tests**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_store_chunk_candidates_are_aggregated_by_problem_and_keep_raw_chunks tests/backend/test_online_retrieval_pipeline.py::test_online_pipeline_accepts_store_injection_and_preserves_debug_outputs -q
```

Expected: PASS.

### Task 4: Honest Graph Paths For Exact Problem Queries

**Files:**
- Modify: `backend/app/retrieval/pipeline.py:205-281`
- Modify: `backend/app/retrieval/pipeline.py:653-673`
- Modify: `backend/app/main.py:298-331`
- Modify: `tests/backend/test_online_retrieval_pipeline.py`
- Modify: `tests/backend/test_analysis.py`

- [ ] **Step 1: Add failing graph exact seed test**

Append to `tests/backend/test_online_retrieval_pipeline.py`:

```python
def test_graph_search_for_exact_problem_returns_problem_node_paths_with_source_labels():
    documents = (_uva_document(),)
    graph_store = InMemoryGraphStore()
    graph_store.upsert_entities(
        (
            EntityRecord(id="uva-10653", name="Bombs! NO they are Mines!!", type="problem"),
            EntityRecord(id="concept:bfs", name="BFS", type="algorithm"),
            EntityRecord(id="concept:queue", name="Queue", type="data_structure"),
        )
    )
    graph_store.upsert_relations(
        (
            RelationRecord(id="uva-10653->concept:bfs", source_id="uva-10653", target_id="concept:bfs", type="REQUIRES", weight=1.0),
            RelationRecord(id="uva-10653->concept:queue", source_id="uva-10653", target_id="concept:queue", type="REQUIRES", weight=1.0),
        )
    )
    understanding = QueryUnderstandingService(documents).understand("UVA-10653 - Bombs! NO they are Mines!!")
    matched = ExactProblemMatcher(documents).match(understanding)
    linked = EntityLinkingService().link(understanding, matched_problem=matched)

    result = GraphSearchService(documents, graph_store=graph_store).search(
        linked,
        matched_problem=matched,
        top_k=3,
    )

    assert result.candidates[0].id == "uva-10653"
    assert result.paths
    assert all(path["pathSource"] == "neo4j" for path in result.paths)
    assert any(path["nodes"] == ["input", "uva-10653", "concept:bfs"] for path in result.paths)
```

Add inferred fallback test:

```python
def test_graph_search_marks_document_concept_fallback_paths_as_inferred():
    documents = (_uva_document(),)
    understanding = QueryUnderstandingService(documents).understand("UVA-10653 - Bombs! NO they are Mines!!")
    matched = ExactProblemMatcher(documents).match(understanding)
    linked = EntityLinkingService().link(understanding, matched_problem=matched)

    result = GraphSearchService(documents).search(linked, matched_problem=matched, top_k=3)

    assert result.candidates[0].id == "uva-10653"
    assert result.paths
    assert all(path["pathSource"] == "inferred" for path in result.paths)
    assert all("storePath" not in path for path in result.paths)
    assert "not returned by Neo4j" in result.paths[0]["rationale"]
```

- [ ] **Step 2: Run the failing graph tests**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_graph_search_for_exact_problem_returns_problem_node_paths_with_source_labels tests/backend/test_online_retrieval_pipeline.py::test_graph_search_marks_document_concept_fallback_paths_as_inferred -q
```

Expected: FAIL because `GraphSearchService.search()` has no `matched_problem` argument and paths have no `pathSource`.

- [ ] **Step 3: Extend graph search signature**

Change:

```python
    def search(self, linked_entities: Sequence[JsonMap], *, top_k: int) -> GraphSearchResult:
```

to:

```python
    def search(
        self,
        linked_entities: Sequence[JsonMap],
        *,
        top_k: int,
        matched_problem: ExactProblemMatch | None = None,
    ) -> GraphSearchResult:
```

Pass `matched_problem` to `_search_store()`.

- [ ] **Step 4: Add inferred path helper**

Add this helper near `_normalize_graph_store_path()`:

```python
def _inferred_problem_paths(document: RetrievalDocument) -> tuple[JsonMap, ...]:
    paths: list[JsonMap] = []
    for concept in document.concepts:
        paths.append(
            {
                "nodes": ["input", document.id, f"concept:{_slug(concept)}"],
                "relations": ["EXACT_MATCH", "REQUIRES"],
                "score": 1.0,
                "pathSource": "inferred",
                "rationale": f"inferred from matched problem {document.title}; not returned by Neo4j",
            }
        )
    if document.problem_type:
        paths.append(
            {
                "nodes": ["input", document.id, f"pattern:{_slug(document.problem_type)}"],
                "relations": ["EXACT_MATCH", "HAS_PATTERN"],
                "score": 1.0,
                "pathSource": "inferred",
                "rationale": f"inferred from matched problem {document.title}; not returned by Neo4j",
            }
        )
    return tuple(paths)
```

- [ ] **Step 5: Add exact problem behavior to in-memory graph search**

At the top of non-store graph search, before linked concept matching:

```python
        if matched_problem is not None:
            matched_document = next((document for document in self._documents if document.id == matched_problem.problem_id), None)
            if matched_document is not None:
                return GraphSearchResult(
                    candidates=(_candidate_from_document(matched_document, source="graph", score=1.0),),
                    paths=_inferred_problem_paths(matched_document),
                )
```

- [ ] **Step 6: Add exact problem behavior to store graph search**

In `_search_store()`, before looping all documents, add:

```python
        if matched_problem is not None:
            matched_document = next((document for document in self._documents if document.id == matched_problem.problem_id), None)
            if matched_document is not None:
                exact_paths: list[JsonMap] = []
                for concept in matched_document.concepts:
                    concept_id = f"concept:{_slug(concept)}"
                    for path in self._graph_store.find_paths(matched_document.id, concept_id, max_hops=3):
                        exact_paths.append(
                            _normalize_graph_store_path(
                                path,
                                entity={"entityId": concept_id, "name": concept},
                                document=matched_document,
                                source_node=matched_document.id,
                                target_node=concept_id,
                            )
                        )
                if matched_document.problem_type:
                    pattern_id = f"pattern:{_slug(matched_document.problem_type)}"
                    for path in self._graph_store.find_paths(matched_document.id, pattern_id, max_hops=3):
                        exact_paths.append(
                            _normalize_graph_store_path(
                                path,
                                entity={"entityId": pattern_id, "name": matched_document.problem_type},
                                document=matched_document,
                                source_node=matched_document.id,
                                target_node=pattern_id,
                            )
                        )
                paths_for_exact = tuple(exact_paths) if exact_paths else _inferred_problem_paths(matched_document)
                return GraphSearchResult(
                    candidates=(_candidate_from_document(matched_document, source="graph", score=1.0),),
                    paths=paths_for_exact,
                )
```

Update `_normalize_graph_store_path()` signature:

```python
def _normalize_graph_store_path(
    path: JsonMap,
    *,
    entity: JsonMap,
    document: RetrievalDocument,
    source_node: str | None = None,
    target_node: str | None = None,
) -> JsonMap:
```

Return:

```python
    return {
        "nodes": ["input", source_node or entity_id, target_node or document.id],
        "relations": ["MENTIONS", "REQUIRED_BY"],
        "score": round(float(path.get("score", 0.0)), 6),
        "pathSource": "neo4j",
        "rationale": f"Neo4j path linked {entity_name} to {document.title}",
        "storePath": {
            "nodes": raw_nodes,
            "relations": raw_relations,
        },
    }
```

- [ ] **Step 7: Pass matched problem from pipeline to graph search**

In `OnlineQueryPipeline.run()`:

```python
        graph_result = GraphSearchService(
            self._documents,
            graph_store=self._graph_store,
        ).search(
            linked_entities,
            top_k=max(top_k * 2, top_k),
            matched_problem=matched_problem,
        )
```

- [ ] **Step 8: Make Graph Evidence and Graph Trace share retrieval graph paths**

In `backend/app/main.py`, add helper:

```python
def _analysis_paths_from_graph_trace(paths: list[dict[str, Any]]) -> list[AnalysisEvidencePathResponse]:
    responses: list[AnalysisEvidencePathResponse] = []
    for index, path in enumerate(paths, start=1):
        nodes = [str(node) for node in path.get("nodes", [])]
        relations = [str(relation) for relation in path.get("relations", [])]
        source = str(path.get("pathSource") or "unknown")
        responses.append(
            AnalysisEvidencePathResponse(
                title=f"Graph path {index} ({source})",
                nodes=[
                    AnalysisEvidenceNodeResponse(
                        id=node,
                        label=node,
                        type="problem" if node.startswith(("uva-", "leetcode-")) or node == "input" else "concept",
                    )
                    for node in nodes
                ],
                edges=[
                    AnalysisEvidenceEdgeResponse(
                        **{
                            "from": nodes[position],
                            "to": nodes[position + 1],
                            "relation": relations[position] if position < len(relations) else "RELATED",
                            "weight": float(path.get("score", 1.0)),
                        }
                    )
                    for position in range(max(len(nodes) - 1, 0))
                ],
            )
        )
    return responses
```

Change the `evidencePaths=` assignment in `AnalysisResponse`:

```python
        evidencePaths=_analysis_paths_from_graph_trace(evidence_bundle.to_mapping()["graphPaths"])
        or [
            AnalysisEvidencePathResponse(
                title=path.title,
                nodes=[
                    AnalysisEvidenceNodeResponse(id=node.id, label=node.label, type=node.type)
                    for node in path.nodes
                ],
                edges=[
                    AnalysisEvidenceEdgeResponse(
                        **{"from": edge.from_id, "to": edge.to, "relation": edge.relation, "weight": edge.weight}
                    )
                    for edge in path.edges
                ],
            )
            for path in result.evidence_paths
        ],
```

This keeps legacy analysis evidence only when retrieval graph trace has no path evidence.

- [ ] **Step 9: Run graph and analysis tests**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_graph_search_for_exact_problem_returns_problem_node_paths_with_source_labels tests/backend/test_online_retrieval_pipeline.py::test_graph_search_marks_document_concept_fallback_paths_as_inferred tests/backend/test_analysis.py -q
```

Expected: PASS after existing graph-path assertions are updated to expect `pathSource`.

### Task 5: Fusion, Reranker, And Evidence Separation

**Files:**
- Modify: `backend/app/retrieval/pipeline.py:284-399`
- Modify: `tests/backend/test_online_retrieval_pipeline.py`

- [ ] **Step 1: Add failing pinning test**

Append to `tests/backend/test_online_retrieval_pipeline.py`:

```python
def test_matched_problem_pin_does_not_change_similar_problem_ranking():
    matched = ExactProblemMatcher((_uva_document(),)).match(
        QueryUnderstandingService((_uva_document(),)).understand("UVA-10653 - Bombs! NO they are Mines!!")
    )
    assert matched is not None
    unrelated = RetrievalCandidate(
        id="leetcode-1091",
        title="Shortest Path in Binary Matrix",
        source="hybrid",
        score=0.80,
        text="BFS shortest path",
        concepts=("BFS", "Queue"),
        problem_type="Graph Traversal",
    )
    weaker = RetrievalCandidate(
        id="leetcode-994",
        title="Rotting Oranges",
        source="hybrid",
        score=0.60,
        text="Multi-source BFS",
        concepts=("BFS", "Queue"),
        problem_type="Graph Traversal",
    )

    evidence = EvidenceBuilder().build((unrelated, matched.candidate, weaker), (), matched_problem=matched)
    evidence_map = evidence.to_mapping()

    assert evidence_map["matchedProblem"]["id"] == "uva-10653"
    assert [problem["id"] for problem in evidence_map["similarProblems"]] == ["leetcode-1091", "leetcode-994"]
```

- [ ] **Step 2: Run failing pinning test**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_matched_problem_pin_does_not_change_similar_problem_ranking -q
```

Expected: FAIL until `EvidenceBuilder` filters exact match from similar problems.

- [ ] **Step 3: Keep exact pin outside `HybridFusionService.fuse()` ranking**

Do not insert `matched_problem.candidate` into `vector_candidates`, `graph_candidates`, `bm25_candidates`, or `fused_candidates` just to pin it. The exact seed is carried by:

```python
matched_problem: ExactProblemMatch | None
```

and emitted through:

```python
trace["matchedProblem"]
evidenceBundle["matchedProblem"]
response["matchedProblem"]
```

If exact match appears naturally through vector, graph, or BM25, it may be part of the source candidates and fusion scores. The implementation must not add an artificial score boost to similar-problem ranking.

- [ ] **Step 4: Reclassify Visited Array as state tracking evidence**

Change `_classify_concept()` in `backend/app/retrieval/pipeline.py`:

```python
def _classify_concept(name: str) -> str:
    lowered = name.lower()
    if lowered in {"bfs", "dfs", "dijkstra", "dynamic programming", "binary search"}:
        return "algorithm"
    if lowered in {"queue", "stack", "heap", "array", "hash map"}:
        return "data_structure"
    if lowered in {"visited array", "visited set", "state tracking"}:
        return "technique"
    return "concept"
```

Update the evidence loop:

```python
        techniques: list[str] = []
        for candidate in (*(([matched_problem.candidate] if matched_problem else [])), *candidates):
            for concept in candidate.concepts:
                kind = _classify_concept(concept)
                if kind == "algorithm":
                    _append_unique(algorithms, concept)
                elif kind == "data_structure":
                    _append_unique(data_structures, concept)
                elif kind == "technique":
                    _append_unique(techniques, concept)
```

Pass `technique_evidence=techniques` to `RetrievalEvidenceBundle`.

- [ ] **Step 5: Localize common mistakes in backend evidence**

Replace:

```python
            common_mistakes=[
                "forget visited state and revisit the same node",
                "enqueue a node before recording its distance or state",
            ],
```

with:

```python
            common_mistakes=[
                "忘記標記 visited，導致同一個節點重複入隊",
                "節點入隊時沒有同步記錄距離或狀態",
            ],
```

- [ ] **Step 6: Run evidence tests**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_matched_problem_pin_does_not_change_similar_problem_ranking tests/backend/test_online_retrieval_pipeline.py::test_evidence_and_context_builders_create_stable_llm_context -q
```

Expected: PASS after changing assertions from `dataStructureEvidence` to `techniqueEvidence` for `Visited Array`.

### Task 6: BM25 Alias Artifact And Runtime Provider Metadata

**Files:**
- Modify: `backend/app/ingestion/pipeline.py:278-309`
- Modify: `backend/app/ingestion/pipeline.py:335-341`
- Modify: `backend/app/providers.py:8-45`
- Modify: `backend/app/retrieval/runtime.py:38-43`
- Modify: `backend/app/retrieval/runtime.py:120-190`
- Modify: `backend/app/main.py:117-157`
- Modify: `tests/backend/test_runtime_retrieval.py`
- Modify: `tests/backend/test_contracts_and_providers.py`

- [ ] **Step 1: Add failing BM25 alias artifact test**

Append to `tests/backend/test_runtime_retrieval.py`:

```python
def test_json_bm25_store_matches_problem_alias_text(tmp_path):
    from backend.app.retrieval.runtime import JsonBM25Store

    index_path = tmp_path / "bm25_index.json"
    index_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "id": "uva-10653:statement:0",
                        "text": "uva-10653 UVa 10653 Bombs! NO they are Mines!! grid bombs bfs",
                        "problemId": "uva-10653",
                        "payload": {
                            "problemId": "uva-10653",
                            "source": "UVa",
                            "sourceId": "10653",
                            "title": "Bombs! NO they are Mines!!",
                            "problemType": "Graph Traversal",
                            "concepts": ["BFS", "Queue", "Visited Array"],
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    store = JsonBM25Store.from_path(index_path)
    results = store.search("UVA-10653 - Bombs! NO they are Mines!!", top_k=1)

    assert results[0].id == "uva-10653:statement:0"
    assert results[0].score > 0
```

- [ ] **Step 2: Add failing provider metadata test**

Append to `tests/backend/test_contracts_and_providers.py`:

```python
def test_deterministic_mock_embedding_provider_declares_provider_kind():
    provider = DeterministicMockEmbeddingProvider(dimension=8)

    assert provider.model_name == "BAAI/bge-m3"
    assert provider.provider_kind == "mock"
```

- [ ] **Step 3: Run failing tests**

Run:

```powershell
python -m pytest tests/backend/test_runtime_retrieval.py::test_json_bm25_store_matches_problem_alias_text tests/backend/test_contracts_and_providers.py::test_deterministic_mock_embedding_provider_declares_provider_kind -q
```

Expected: provider test FAIL until `provider_kind` exists. The BM25 test may already pass because the test writes alias text directly; artifact generation is covered in the next step.

- [ ] **Step 4: Enrich BM25 artifact generation**

In `backend/app/ingestion/pipeline.py`, add:

```python
def _bm25_search_text(chunk: ProblemChunk) -> str:
    alias_text = " ".join(
        value
        for value in (
            chunk.problem_id,
            chunk.source,
            chunk.source_id,
            f"{chunk.source}-{chunk.source_id}",
            f"{chunk.source} {chunk.source_id}",
            chunk.title,
            chunk.problem_type,
            " ".join(chunk.concepts),
        )
        if value
    )
    return _clean_text(f"{alias_text} {chunk.text}")
```

Change `_write_bm25_index()`:

```python
def _write_bm25_index(path: Path, chunks: tuple[ProblemChunk, ...]) -> None:
    documents = [
        {
            "id": chunk.id,
            "text": _bm25_search_text(chunk),
            "problemId": chunk.problem_id,
            "tokens": _tokens(_bm25_search_text(chunk)),
            "payload": chunk.to_mapping(),
        }
        for chunk in chunks
    ]
    _write_json(path, {"documents": documents})
```

This makes title/source aliases searchable but does not classify title aliases as exact. Exact title and partial title remain the responsibility of `ExactProblemMatcher`.

- [ ] **Step 5: Reclassify Visited Array in ingestion artifacts**

Change `_classify_concept()` in `backend/app/ingestion/pipeline.py`:

```python
def _classify_concept(name: str) -> str:
    lowered = name.lower()
    if lowered in {"bfs", "dfs", "dijkstra", "binary search", "dynamic programming"}:
        return "algorithm"
    if lowered in {"queue", "stack", "heap", "array", "hash map"}:
        return "data_structure"
    if lowered in {"visited array", "visited set", "state tracking"}:
        return "technique"
    return "concept"
```

- [ ] **Step 6: Add provider kind to mock provider**

In `backend/app/providers.py`, update the protocol:

```python
class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    def provider_kind(self) -> str:
        raise NotImplementedError
```

Add to `DeterministicMockEmbeddingProvider`:

```python
    @property
    def provider_kind(self) -> str:
        return "mock"
```

- [ ] **Step 7: Expose runtime provider metadata**

In `backend/app/retrieval/runtime.py`, change `RuntimeRetrieval`:

```python
@dataclass(frozen=True)
class RuntimeRetrieval:
    backend: RetrievalBackend
    pipeline: OnlineQueryPipeline
    candidate_sources: dict[str, str]
    provider_sources: dict[str, JsonMap]
```

For local mode:

```python
            provider_sources={
                "embedding": {
                    "provider": getattr(embedding_provider or DeterministicMockEmbeddingProvider(), "provider_kind", "mock"),
                    "model": (embedding_provider or DeterministicMockEmbeddingProvider()).model_name,
                },
                "reranker": {"provider": "mock", "model": "BAAI/bge-reranker-v2-m3"},
            },
```

For stores mode, reuse the resolved provider instance rather than constructing it twice:

```python
        resolved_embedding_provider = embedding_provider or DeterministicMockEmbeddingProvider()
```

Pass it to `OnlineQueryPipeline` and set:

```python
            provider_sources={
                "embedding": {
                    "provider": getattr(resolved_embedding_provider, "provider_kind", "unknown"),
                    "model": resolved_embedding_provider.model_name,
                    "adapter": "qdrant",
                },
                "reranker": {"provider": "mock", "model": "BAAI/bge-reranker-v2-m3"},
            },
```

Update `add_runtime_debug_trace()` signature:

```python
def add_runtime_debug_trace(
    trace: JsonMap,
    candidate_sources: Mapping[str, str],
    provider_sources: Mapping[str, JsonMap] | None = None,
) -> JsonMap:
```

Add:

```python
    if provider_sources:
        labeled["providerSources"] = {key: dict(value) for key, value in provider_sources.items()}
```

- [ ] **Step 8: Expose provider metadata in API response**

In `backend/app/main.py`, extend `RetrievalConfigResponse`:

```python
class ProviderDescriptorResponse(BaseModel):
    provider: str
    model: str
    adapter: str | None = None


class RetrievalConfigResponse(BaseModel):
    embeddingModel: str
    rerankerModel: str
    language: str
    embeddingProvider: ProviderDescriptorResponse | None = None
    rerankerProvider: ProviderDescriptorResponse | None = None
```

When building response:

```python
        retrievalConfig=RetrievalConfigResponse(
            embeddingModel=result.retrieval_config.embedding_model,
            rerankerModel=result.retrieval_config.reranker_model,
            language=result.retrieval_config.language,
            embeddingProvider=runtime_retrieval.provider_sources.get("embedding"),
            rerankerProvider=runtime_retrieval.provider_sources.get("reranker"),
        ),
```

When debug trace is added:

```python
        retrieval_trace = add_runtime_debug_trace(
            retrieval_trace,
            runtime_retrieval.candidate_sources,
            runtime_retrieval.provider_sources,
        )
```

- [ ] **Step 9: Run runtime/provider tests**

Run:

```powershell
python -m pytest tests/backend/test_runtime_retrieval.py tests/backend/test_contracts_and_providers.py -q
```

Expected: PASS after updating existing `RuntimeRetrieval(...)` test fixtures to include `provider_sources`.

### Task 7: Context Builder And API Matched Problem Output

**Files:**
- Modify: `backend/app/retrieval/pipeline.py:402-452`
- Modify: `backend/app/main.py:101-157`
- Modify: `tests/backend/test_online_retrieval_pipeline.py`
- Modify: `tests/backend/test_analysis.py`

- [ ] **Step 1: Add failing localized context test**

Replace English context assertions in `tests/backend/test_online_retrieval_pipeline.py::test_context_builder_includes_enriched_candidate_evidence` with:

```python
    assert "查詢理解" in context
    assert "相似題" in context
    assert "答案摘要: Use BFS from all rotten oranges." in context
    assert "解題提示: Push all rotten oranges first." in context
    assert "難度: Medium" in context
    assert "限制: 1 <= m, n <= 10" in context
    assert "圖路徑" in context
```

Add matched-problem context test:

```python
def test_context_builder_includes_matched_problem_separately():
    matched = ExactProblemMatcher((_uva_document(),)).match(
        QueryUnderstandingService((_uva_document(),)).understand("UVA-10653 - Bombs! NO they are Mines!!")
    )
    assert matched is not None

    evidence = EvidenceBuilder().build((), (), matched_problem=matched)
    context = ContextBuilder().build(
        QueryUnderstandingService((_uva_document(),)).understand("UVA-10653 - Bombs! NO they are Mines!!"),
        evidence,
    )

    assert "命中題目" in context
    assert "uva-10653 Bombs! NO they are Mines!!" in context
    assert "相似題" in context
```

- [ ] **Step 2: Run failing context tests**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py::test_context_builder_includes_enriched_candidate_evidence tests/backend/test_online_retrieval_pipeline.py::test_context_builder_includes_matched_problem_separately -q
```

Expected: FAIL while context labels are still English.

- [ ] **Step 3: Localize `ContextBuilder`**

Replace the initial `lines` list:

```python
        lines = [
            "查詢理解",
            f"- 意圖: {query_understanding.intent}",
            f"- 輸入類型: {query_understanding.input_kind}",
            f"- 關鍵詞: {', '.join(query_understanding.keywords)}",
            "",
            "命中題目",
        ]
        matched = evidence.get("matchedProblem")
        if matched:
            lines.append(
                f"- {matched['id']} {matched['title']} "
                f"(matchKind={matched['matchKind']}, confidence={matched['confidence']})"
            )
            if matched.get("answerHint"):
                lines.append(f"  答案摘要: {matched['answerHint']}")
            for hint in matched.get("solutionHints", []):
                lines.append(f"  解題提示: {hint}")
        else:
            lines.append("- 無")
        lines.extend(["", "相似題"])
```

Replace detail labels in the similar-problem loop:

```python
            if problem.get("answerHint"):
                lines.append(f"  答案摘要: {problem['answerHint']}")
            for hint in problem.get("solutionHints", []):
                lines.append(f"  解題提示: {hint}")
            if problem.get("difficulty"):
                lines.append(f"  難度: {problem['difficulty']}")
            if problem.get("constraints"):
                lines.append(f"  限制: {', '.join(problem['constraints'])}")
        lines.extend(["", "圖路徑"])
```

Replace final labels:

```python
                "演算法證據",
                f"- {', '.join(evidence['algorithmEvidence'])}",
                "資料結構證據",
                f"- {', '.join(evidence['dataStructureEvidence'])}",
                "技巧證據",
                f"- {', '.join(evidence.get('techniqueEvidence', []))}",
                "題型證據",
                f"- {', '.join(evidence['patternEvidence'])}",
                "常見錯誤",
                *[f"- {mistake}" for mistake in evidence["commonMistakes"]],
```

- [ ] **Step 4: Add top-level `matchedProblem` response**

In `backend/app/main.py`, add:

```python
class MatchedProblemResponse(BaseModel):
    id: str
    title: str
    source: str
    sourceId: str
    matchKind: str
    confidence: float
    score: float
    sharedConcepts: list[str]
    problemType: str
    answerHint: str | None = None
    solutionHints: list[str] = []
    difficulty: str | None = None
    constraints: list[str] = []
```

Add to `AnalysisResponse`:

```python
    matchedProblem: MatchedProblemResponse | None = None
```

In response construction:

```python
        matchedProblem=(
            MatchedProblemResponse(**pipeline_result.matched_problem.to_mapping())
            if pipeline_result.matched_problem
            else None
        ),
```

- [ ] **Step 5: Add API test for top-level matched problem**

In `tests/backend/test_analysis.py`, add:

```python
def test_analysis_exact_problem_query_exposes_matched_problem_separately():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "UVA-10653 - Bombs! NO they are Mines!!"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["inputKind"] == "problem"
    assert payload["matchedProblem"]["id"] == "uva-10653"
    assert payload["retrievalTrace"]["matchedProblem"]["id"] == "uva-10653"
    assert payload["evidenceBundle"]["matchedProblem"]["id"] == "uva-10653"
    assert all(problem["id"] != "uva-10653" for problem in payload["evidenceBundle"]["similarProblems"])
```

- [ ] **Step 6: Run API and context tests**

Run:

```powershell
python -m pytest tests/backend/test_analysis.py tests/backend/test_online_retrieval_pipeline.py -q
```

Expected: PASS after existing English context assertions are updated to the new Traditional Chinese labels.

### Task 8: Frontend Contract And Display

**Files:**
- Modify: `frontend/src/types.ts:47-112`
- Modify: `frontend/src/api.ts:19-136`
- Modify: `frontend/src/api.ts:303-413`
- Modify: `frontend/src/App.tsx:240-290`
- Modify: `frontend/src/App.tsx:310-438`
- Modify: `frontend/src/App.tsx:503-630`

- [ ] **Step 1: Add frontend types**

In `frontend/src/types.ts`, add:

```ts
export interface MatchedProblem {
  id: string;
  title: string;
  source: string;
  sourceId: string;
  matchKind: string;
  confidence: number;
  score: number;
  sharedConcepts: string[];
  problemType: string;
  answerHint?: string;
  solutionHints?: string[];
  difficulty?: string;
  constraints?: string[];
}

export interface ProviderDescriptor {
  provider: string;
  model: string;
  adapter?: string;
}
```

Extend `RetrievalConfig`:

```ts
export interface RetrievalConfig {
  embeddingModel: string;
  rerankerModel: string;
  language: string;
  embeddingProvider?: ProviderDescriptor;
  rerankerProvider?: ProviderDescriptor;
}
```

Extend `TraceCandidate`:

```ts
  rawChunks?: TraceCandidate[];
```

Extend `GraphPathTrace`:

```ts
  pathSource?: "neo4j" | "inferred" | string;
```

Extend `RetrievalTrace`, `EvidenceBundle`, and `AnalysisResponse`:

```ts
  matchedProblem?: MatchedProblem;
```

- [ ] **Step 2: Normalize matched problem and provider data**

In `frontend/src/api.ts`, add:

```ts
function normalizeMatchedProblem(value: unknown): MatchedProblem | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  const id = asString(record.id ?? record.problemId ?? record.problem_id, "");
  if (!id) {
    return undefined;
  }
  return {
    id,
    title: asString(record.title, id),
    source: asString(record.source, ""),
    sourceId: asString(record.sourceId ?? record.source_id, ""),
    matchKind: asString(record.matchKind ?? record.match_kind, ""),
    confidence: asNumber(record.confidence, 0),
    score: asNumber(record.score, 0),
    sharedConcepts: asStringArray(record.sharedConcepts ?? record.shared_concepts ?? record.concepts),
    problemType: asString(record.problemType ?? record.problem_type, ""),
    answerHint: asString(record.answerHint ?? record.answer_hint, ""),
    solutionHints: asStringArray(record.solutionHints ?? record.solution_hints),
    difficulty: asString(record.difficulty, ""),
    constraints: asStringArray(record.constraints)
  };
}

function normalizeProviderDescriptor(value: unknown): ProviderDescriptor | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  return {
    provider: asString(record.provider, ""),
    model: asString(record.model, ""),
    adapter: asString(record.adapter, "")
  };
}
```

In `normalizeCandidate()`, preserve raw chunks:

```ts
    rawChunks: pickArray(record.payload, ["rawChunks", "raw_chunks"])
      .map(normalizeCandidate)
      .filter((candidate): candidate is TraceCandidate => candidate !== null && candidate.id.length > 0),
```

If `record.payload` is not directly usable by `pickArray`, use:

```ts
  const payload = asRecord(record.payload) ?? undefined;
```

and read raw chunks from `payload`.

In `normalizeTrace()`:

```ts
    matchedProblem: normalizeMatchedProblem(record.matchedProblem ?? record.matched_problem),
```

In `normalizeEvidenceBundle()`:

```ts
    matchedProblem: normalizeMatchedProblem(record.matchedProblem ?? record.matched_problem),
```

In `normalizeRetrievalConfig()`:

```ts
    embeddingProvider: normalizeProviderDescriptor(record.embeddingProvider ?? record.embedding_provider),
    rerankerProvider: normalizeProviderDescriptor(record.rerankerProvider ?? record.reranker_provider)
```

In `normalizeResponse()`:

```ts
    matchedProblem: normalizeMatchedProblem(record.matchedProblem ?? record.matched_problem),
```

- [ ] **Step 3: Add Matched Problem panel**

In `frontend/src/App.tsx`, render the panel before retrieval details:

```tsx
<MatchedProblemPanel problem={response.matchedProblem ?? response.evidenceBundle?.matchedProblem ?? response.retrievalTrace?.matchedProblem} />
```

Add component:

```tsx
function MatchedProblemPanel({ problem }: { problem?: MatchedProblem }) {
  if (!problem) {
    return null;
  }

  return (
    <OutputBlock title="命中題目">
      <article className="matched-problem">
        <div className="candidate-main">
          <strong>{problem.sourceId ? `${problem.sourceId} - ${problem.title}` : problem.title}</strong>
          <span>
            {[problem.id, problem.source, problem.matchKind, `信心 ${formatScore(problem.confidence)}`]
              .filter(Boolean)
              .join(" / ")}
          </span>
          {problem.sharedConcepts.length > 0 && <span>{problem.sharedConcepts.join(", ")}</span>}
          {problem.answerHint && <small>{problem.answerHint}</small>}
        </div>
      </article>
    </OutputBlock>
  );
}
```

Update imports to include `MatchedProblem`.

- [ ] **Step 4: Fix CandidateList keys and raw chunk display**

Replace:

```tsx
            <li key={`${title}-${candidate.id}`}>
```

with:

```tsx
            <li key={`${title}-${candidate.id}-${asText(candidate.payload?.storeCandidateId) || candidate.rawChunks?.map((chunk) => asText(chunk.payload?.storeCandidateId)).join("|") || index}`}>
```

Update the map callback:

```tsx
          {candidates.slice(0, 4).map((candidate, index) => (
```

Add raw chunk details under payload:

```tsx
                {candidate.rawChunks && candidate.rawChunks.length > 0 && (
                  <details className="payload-details">
                    <summary>原始 chunks ({candidate.rawChunks.length})</summary>
                    <JsonBlock value={candidate.rawChunks} />
                  </details>
                )}
```

- [ ] **Step 5: Localize visible debug labels**

Change these component labels in `frontend/src/App.tsx`:

```tsx
<span>意圖</span>
<span>輸入類型</span>
<span>關鍵詞</span>
<CandidateList title="向量搜尋 / Qdrant" candidates={trace?.vectorCandidates ?? []} />
<CandidateList title="圖搜尋 / Neo4j" candidates={trace?.graphCandidates ?? []} />
<CandidateList title="BM25 關鍵字搜尋" candidates={trace?.bm25Candidates ?? []} />
<OutputBlock title="混合融合 / 重排序">
<CandidateList title="融合分數" candidates={trace?.fusionScores ?? []} />
<CandidateList title="重排序分數" candidates={trace?.rerankerScores ?? []} />
<OutputBlock title="證據整理">
<EvidenceList title="演算法" items={evidence.algorithmEvidence} />
<EvidenceList title="資料結構" items={evidence.dataStructureEvidence} />
<EvidenceList title="技巧 / 狀態追蹤" items={evidence.techniqueEvidence ?? []} />
<EvidenceList title="題型" items={evidence.patternEvidence} />
<EvidenceList title="常見錯誤" items={evidence.commonMistakes} />
```

Change graph path labels:

```tsx
<p className="eyebrow">圖證據</p>
<h2>Graph Evidence</h2>
<p className="eyebrow">圖路徑追蹤</p>
<h2>Graph Paths</h2>
<p className="muted">沒有圖路徑。</p>
<dt>節點</dt>
<dt>關係</dt>
<dt>依據</dt>
<dt>分數</dt>
<dt>來源</dt>
<summary>Store path</summary>
```

For path source display:

```tsx
              <div>
                <dt>來源</dt>
                <dd>{path.pathSource === "inferred" ? "推論 fallback" : path.pathSource || "-"}</dd>
              </div>
```

- [ ] **Step 6: Show mock provider explicitly**

In `ModelPanel`, replace `dd` values with provider-aware labels:

```tsx
          <dt>Embedding</dt>
          <dd>
            {config?.embeddingProvider
              ? `${config.embeddingProvider.provider} / ${config.embeddingProvider.adapter || "local"} / ${config.embeddingProvider.model}`
              : config?.embeddingModel ?? "BAAI/bge-m3"}
          </dd>
```

```tsx
          <dt>Reranker</dt>
          <dd>
            {config?.rerankerProvider
              ? `${config.rerankerProvider.provider} / ${config.rerankerProvider.model}`
              : config?.rerankerModel ?? "BAAI/bge-reranker-v2-m3"}
          </dd>
```

- [ ] **Step 7: Run frontend build**

Run:

```powershell
npm.cmd run build
```

Expected: PASS with no TypeScript errors.

### Task 9: End-To-End Verification For Original Bug List

**Files:**
- No source edits unless verification exposes a missed requirement.

- [ ] **Step 1: Run backend focused tests**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py tests/backend/test_runtime_retrieval.py tests/backend/test_analysis.py tests/backend/test_contracts_and_providers.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full backend checks**

Run:

```powershell
python -m ruff check .
python -m pytest tests/backend -q
```

Expected: PASS.

- [ ] **Step 3: Run frontend build**

Run:

```powershell
npm.cmd run build
```

Expected: PASS.

- [ ] **Step 4: Run static stores preflight**

Run:

```powershell
.\scripts\quick-start.ps1 -Check -Stores
```

Expected:

```text
Retrieval backend: stores
BM25 index: C:\knowledgegraph_rag\data\processed\bm25_index.json
Processed problems: C:\knowledgegraph_rag\data\processed\problems.json
```

- [ ] **Step 5: Rebuild processed artifacts**

Run the ingestion command documented in `README.md` and `docs/api.md`:

```powershell
python -m backend.app.ingestion build --input data/raw --processed data/processed --target all
```

Expected: command exits with code 0 and these files exist after the run:

```text
data/processed/bm25_index.json
data/processed/qdrant_vectors.json
data/processed/neo4j_graph.json
```

- [ ] **Step 6: Verify live API exact query**

With backend running in stores mode, run:

```powershell
$body = @{ input = "UVA-10653 - Bombs! NO they are Mines!!" } | ConvertTo-Json
$response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/analysis?debug=true" -ContentType "application/json" -Body $body
$response.inputKind
$response.matchedProblem.id
$response.retrievalTrace.matchedProblem.matchKind
$response.evidenceBundle.similarProblems.id
$response.evidenceBundle.graphPaths | Select-Object -First 5
```

Expected:

```text
problem
uva-10653
exact_problem_id
```

The `similarProblems.id` output must not include `uva-10653`. Each graph path must include `pathSource`; real Neo4j paths use `neo4j`, fallback paths use `inferred`.

- [ ] **Step 7: Verify BM25 and vector debug shape**

Run:

```powershell
$response.retrievalTrace.bm25Candidates | Select-Object id,score
$response.retrievalTrace.vectorCandidates | Select-Object id,score
$response.retrievalTrace.vectorCandidates[0].payload.rawChunks | Select-Object id,score
```

Expected:

```text
id        score
--        -----
uva-10653 > 0
```

Vector candidates must show each problem once in the visible list while `payload.rawChunks` preserves the original chunk hits.

- [ ] **Step 8: Verify provider metadata**

Run:

```powershell
$response.retrievalConfig.embeddingProvider
$response.retrievalConfig.rerankerProvider
$response.retrievalTrace.providerSources
```

Expected: provider fields explicitly contain `mock` when deterministic mock providers are in use. The UI must not present only `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` without provider kind.

- [ ] **Step 9: Browser verification**

Open `http://127.0.0.1:5173`, submit:

```text
UVA-10653 - Bombs! NO they are Mines!!
```

Expected visible behavior:

- Query Understanding shows `輸入類型: problem`.
- A separate `命中題目` panel shows `10653 - Bombs! NO they are Mines!!`.
- Vector, Graph, BM25, Fusion, and Reranker panels remain visible and do not need to pin `uva-10653` into their ranking to show the match.
- Similar Problems does not include the matched problem.
- Graph Evidence and Graph Paths agree on the same path set.
- Inferred paths are labeled as inferred fallback and do not imply Neo4j returned them.
- Provider / Adapter shows mock versus real provider kind.
- Visible labels are Traditional Chinese except product names, protocol names, and model names.
- Browser console has no duplicate-key warnings for vector or BM25 candidates.

- [ ] **Step 10: Cleanup check**

Run:

```powershell
git diff --check
git status --short --branch
```

Expected: no whitespace errors. `git status` should show only files intentionally modified by the implementation and any pre-existing untracked `docs/superpowers/` state.

## Requirement Coverage Checklist

- ExactProblemMatcher as independent retrieval seed: Tasks 1 and 2.
- Exact title versus partial title semantics: Task 1.
- Matched Problem independent of retrieval mode and visible across Vector/Graph/Hybrid UI: Tasks 2, 7, and 8.
- Graph Evidence must not pretend fallback is Neo4j: Task 4.
- Chunk aggregation preserves `rawChunks`: Task 3 and Task 8.
- Pinning only applies to Matched Problem, not Similar Problems ranking: Task 5.
- BM25 exact query no longer shows all-zero unrelated candidates: Tasks 3 and 6.
- Query Understanding identifies UVA exact problem queries as `problem`: Tasks 1 and 2.
- Graph Search, Graph Evidence, and Graph Trace consistency: Task 4.
- Vector duplicate chunk display fixed while debug remains inspectable: Tasks 3 and 8.
- Provider/Adapter marks mock versus real providers: Task 6 and Task 8.
- UI/debug labels localized to Traditional Chinese: Task 5, Task 7, and Task 8.

## Self-Review Notes

- This plan does not rely on score boosting to hide retrieval failures. Exact matches are carried as `matchedProblem`, and retrieval rankings remain observable.
- `rawChunks` is intentionally retained in backend trace and frontend payload details so debugging can still inspect original Qdrant/BM25 chunk hits.
- `pathSource` is mandatory for graph paths. `storePath` appears only for real store-returned paths.
- Existing API fields stay backward compatible: `similarProblems`, `retrievalTrace`, `evidenceBundle`, and `contextPreview` remain present. New fields are additive.

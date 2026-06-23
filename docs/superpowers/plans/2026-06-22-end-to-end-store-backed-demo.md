# End-to-End Store-Backed Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the FastAPI `POST /api/analysis?debug=true` runtime choose between the existing local fallback and real Qdrant + Neo4j + BM25 index stores.

**Architecture:** Add a small runtime retrieval factory that reads environment-backed settings, builds either a local `OnlineQueryPipeline()` or a store-injected `OnlineQueryPipeline`, and exposes debug labels for candidate source provenance. Keep the retrieval services and store adapters already added in the previous phase; this phase wires stores into the existing FastAPI runtime and keeps the current runtime documents as the candidate/document universe.

**Tech Stack:** Python 3.11, FastAPI, pytest, qdrant-client, neo4j Python driver, existing `BM25Store` protocol, PowerShell 5.1-compatible quick-start script, Docker Compose, Vite frontend build.

---

## Scope Decisions

- This phase is runtime wiring only. Do not expand it into a full dataset source replacement.
- Default runtime mode is `RETRIEVAL_BACKEND=local`; it must not start Docker or require Qdrant, Neo4j, or `data/processed/bm25_index.json`.
- Store-backed runtime mode is `RETRIEVAL_BACKEND=stores`; it constructs Qdrant, Neo4j, and BM25 store adapters and injects them into `OnlineQueryPipeline`.
- Graph search may continue to use the existing runtime documents as the candidate set. Do not load documents from `data/processed/problems.json` in this phase; if that becomes necessary, create a separate next-phase plan.
- Store candidates from Qdrant and BM25 may be chunk-level hits and may omit `answer` / `solutionHints`. Do not change the ingestion schema to backfill those fields in this phase.
- The BM25 index file can fail while building the runtime retrieval object because the file is read immediately. Qdrant and Neo4j constructors do not necessarily probe service health, so unavailable services may fail on first query unless a separate health-check step is added.
- Runtime store creation belongs in a focused backend module, not inline in the request handler.
- `RetrievalCandidate.source` remains the logical lane name: `vector`, `graph`, `bm25`, or `hybrid`. Debug provenance is added separately as `candidateSource` on trace candidates and `candidateSources` on the trace.
- `retrievalBackend` is included only when `debug=true`, matching the requested debug-mode response expansion.
- The quick-start default remains local. The full store-backed demo path is `.\scripts\quick-start.ps1 -Stores`.
- Keep `scripts/quick-start.ps1` ASCII-only because this repo has prior Windows PowerShell encoding sensitivity.
- `app.on_event("startup")` is acceptable for this phase. Do not refactor to FastAPI lifespan here; leave that as a future cleanup.
- Do not move to the next task until that task's listed verification command has passed or the expected failure has been recorded.

## File Structure

- Create: `backend/app/retrieval/runtime.py`
  - Owns `RuntimeRetrievalSettings`, env parsing, `JsonBM25Store`, runtime pipeline construction, and debug trace labeling.
  - Does not load `data/processed/problems.json`; `OnlineQueryPipeline` continues using its existing default documents unless tests explicitly inject a small fixture document set.
- Modify: `backend/app/main.py`
  - Builds the runtime retrieval object at startup and lazily on first request for tests that do not run lifespan hooks.
  - Uses the configured pipeline in `/api/analysis`.
  - Adds `retrievalBackend` and debug candidate source labels when `debug=true`.
- Modify: `tests/backend/test_runtime_retrieval.py`
  - Covers local mode without Docker, env parsing, BM25 JSON loading, debug trace labeling, and store-mode injection with fake stores.
- Modify: `tests/backend/test_analysis.py`
  - Extends existing API debug coverage for `retrievalBackend` and `candidateSource` labels.
- Modify: `.env.example`
  - Adds `RETRIEVAL_BACKEND` and `BM25_INDEX_PATH`.
  - Aligns `QDRANT_COLLECTION` with the current ingestion adapter default, `programming_chunks`.
- Modify: `scripts/quick-start.ps1`
  - Adds `-Stores`, `-SkipDocker`, and `-SkipIngestion`.
  - Starts Docker services, runs ingestion, and starts backend/frontend with store env vars when `-Stores` is used.
- Modify: `pyproject.toml`
  - Adds a `docker` pytest marker if the optional integration smoke test is added.
- Create: `tests/backend/test_store_runtime_integration.py`
  - Optional Docker-backed smoke test, skipped unless `RUN_DOCKER_TESTS=1`.
- Modify: `README.md`, `docs/architecture.md`, `docs/api.md`, `scripts/README.md`
  - Documents local vs stores mode, debug response fields, quick-start usage, and verification commands.

---

## Task Completion Rule

Each task has its own verification command. Run that command before starting the next task, and keep the result in the execution notes. The final verification task is a regression sweep, not a substitute for per-task checks.

---

### Task 1: Add Runtime Retrieval Tests First

**Files:**
- Create: `tests/backend/test_runtime_retrieval.py`

- [ ] **Step 1: Create failing runtime retrieval tests**

Create `tests/backend/test_runtime_retrieval.py` with this content:

```python
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from backend.app.providers import DeterministicMockEmbeddingProvider
from backend.app.retrieval.pipeline import RetrievalDocument
from backend.app.stores import BM25Document, SearchCandidate


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


def _write_bm25_index(path: Path) -> None:
    path.write_text(
        """
{
  "documents": [
    {
      "id": "leetcode-994:statement:0",
      "text": "Multi-source BFS with a queue on a grid.",
      "problemId": "leetcode-994",
      "payload": {
        "problemId": "leetcode-994",
        "kind": "statement",
        "text": "Multi-source BFS with a queue on a grid.",
        "concepts": ["BFS", "Queue"],
        "metadata": {
          "source": "LeetCode",
          "sourceId": "994",
          "title": "Rotting Oranges",
          "problemType": "Graph Traversal"
        }
      }
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )


class FakeVectorStore:
    def __init__(self, *, client=None, collection_name: str, url: str, timeout: float = 1.0):
        self.collection_name = collection_name
        self.url = url

    def upsert(self, records: Sequence[object]) -> None:
        raise AssertionError("runtime search should not upsert vector records")

    def search(self, query_vector: Sequence[float], *, top_k: int, filters=None):
        return (
            SearchCandidate(
                id="leetcode-994:statement:0",
                score=0.99,
                payload={
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
                },
            ),
        )


class FakeGraphStore:
    def __init__(self, *, driver=None, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user

    def upsert_entities(self, entities: Sequence[object]) -> None:
        raise AssertionError("runtime search should not upsert graph entities")

    def upsert_relations(self, relations: Sequence[object]) -> None:
        raise AssertionError("runtime search should not upsert graph relations")

    def find_paths(self, source_id: str, target_id: str, *, max_hops: int = 3):
        if source_id == "leetcode-994" and target_id in {"concept:bfs", "concept:queue"}:
            return (
                {
                    "nodes": [source_id, target_id],
                    "relations": ["REQUIRES"],
                    "score": 1.0,
                },
            )
        return ()


def test_load_runtime_retrieval_settings_defaults_to_local():
    from backend.app.retrieval.runtime import load_runtime_retrieval_settings

    settings = load_runtime_retrieval_settings({})

    assert settings.backend == "local"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "programming_chunks"
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password == "password"
    assert settings.bm25_index_path.name == "bm25_index.json"


def test_load_runtime_retrieval_settings_rejects_unknown_backend():
    from backend.app.retrieval.runtime import RuntimeRetrievalError, load_runtime_retrieval_settings

    with pytest.raises(RuntimeRetrievalError, match="unsupported RETRIEVAL_BACKEND"):
        load_runtime_retrieval_settings({"RETRIEVAL_BACKEND": "remote"})


def test_json_bm25_store_loads_processed_index(tmp_path):
    from backend.app.retrieval.runtime import JsonBM25Store

    index_path = tmp_path / "bm25_index.json"
    _write_bm25_index(index_path)

    store = JsonBM25Store.from_path(index_path)
    results = store.search("BFS queue shortest path", top_k=1)

    assert results[0].id == "leetcode-994:statement:0"
    assert results[0].payload["problemId"] == "leetcode-994"
    assert results[0].payload["metadata"]["title"] == "Rotting Oranges"


def test_json_bm25_store_can_accept_runtime_documents_after_loading(tmp_path):
    from backend.app.retrieval.runtime import JsonBM25Store

    index_path = tmp_path / "bm25_index.json"
    _write_bm25_index(index_path)
    store = JsonBM25Store.from_path(index_path)
    store.index_documents((BM25Document(id="extra", text="binary search", payload={"problemId": "extra"}),))

    results = store.search("binary search", top_k=1)

    assert results[0].id == "extra"
    assert results[0].payload["problemId"] == "extra"


def test_build_runtime_retrieval_local_does_not_construct_external_stores(monkeypatch):
    from backend.app.retrieval import runtime

    def fail_qdrant(**kwargs):
        raise AssertionError("local mode must not construct Qdrant")

    def fail_neo4j(**kwargs):
        raise AssertionError("local mode must not construct Neo4j")

    monkeypatch.setattr(runtime, "QdrantVectorStore", fail_qdrant)
    monkeypatch.setattr(runtime, "Neo4jGraphStore", fail_neo4j)
    settings = runtime.RuntimeRetrievalSettings(backend="local")

    configured = runtime.build_runtime_retrieval(
        settings=settings,
        documents=_documents(),
        embedding_provider=DeterministicMockEmbeddingProvider(dimension=8),
    )
    result = configured.pipeline.run("BFS queue shortest path", top_k=2)

    assert configured.backend == "local"
    assert configured.candidate_sources == {"vector": "local", "graph": "local", "bm25": "local"}
    assert result.vector_candidates
    assert result.bm25_candidates


def test_build_runtime_retrieval_stores_injects_qdrant_neo4j_and_bm25(monkeypatch, tmp_path):
    from backend.app.retrieval import runtime

    index_path = tmp_path / "bm25_index.json"
    _write_bm25_index(index_path)
    monkeypatch.setattr(runtime, "QdrantVectorStore", FakeVectorStore)
    monkeypatch.setattr(runtime, "Neo4jGraphStore", FakeGraphStore)
    settings = runtime.RuntimeRetrievalSettings(
        backend="stores",
        qdrant_url="http://qdrant.example:6333",
        qdrant_collection="programming_chunks",
        neo4j_uri="bolt://neo4j.example:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        bm25_index_path=index_path,
    )

    configured = runtime.build_runtime_retrieval(
        settings=settings,
        documents=_documents(),
        embedding_provider=DeterministicMockEmbeddingProvider(dimension=8),
    )
    result = configured.pipeline.run("BFS queue graph traversal", top_k=2)
    trace = result.trace.to_mapping()

    assert configured.backend == "stores"
    assert configured.candidate_sources == {
        "vector": "qdrant",
        "graph": "neo4j",
        "bm25": "bm25_index",
    }
    assert trace["vectorCandidates"][0]["payload"]["storeCandidateId"] == "leetcode-994:statement:0"
    assert trace["bm25Candidates"][0]["payload"]["storeCandidateId"] == "leetcode-994:statement:0"
    assert trace["vectorCandidates"][0]["payload"]["answer"] == ""
    assert trace["bm25Candidates"][0]["payload"]["answer"] == ""
    assert trace["graphCandidates"][0]["id"] == "leetcode-994"
    assert any("storePath" in path for path in result.graph_paths)


def test_add_runtime_debug_trace_labels_candidate_sources():
    from backend.app.retrieval.runtime import add_runtime_debug_trace

    trace = {
        "vectorCandidates": [{"id": "v"}],
        "graphCandidates": [{"id": "g"}],
        "bm25Candidates": [{"id": "b"}],
        "fusionScores": [{"id": "h"}],
        "rerankerScores": [{"id": "r"}],
    }

    labeled = add_runtime_debug_trace(
        trace,
        {"vector": "qdrant", "graph": "neo4j", "bm25": "bm25_index"},
    )

    assert labeled["candidateSources"] == {
        "vector": "qdrant",
        "graph": "neo4j",
        "bm25": "bm25_index",
    }
    assert labeled["vectorCandidates"][0]["candidateSource"] == "qdrant"
    assert labeled["graphCandidates"][0]["candidateSource"] == "neo4j"
    assert labeled["bm25Candidates"][0]["candidateSource"] == "bm25_index"
    assert labeled["fusionScores"][0]["candidateSource"] == "hybrid"
    assert labeled["rerankerScores"][0]["candidateSource"] == "hybrid"
```

- [ ] **Step 2: Run the new tests and confirm the missing module failure**

Run:

```powershell
python -m pytest tests/backend/test_runtime_retrieval.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.app.retrieval.runtime'`.

---

### Task 2: Implement Runtime Retrieval Factory

**Files:**
- Create: `backend/app/retrieval/runtime.py`
- Test: `tests/backend/test_runtime_retrieval.py`

- [ ] **Step 1: Create the runtime retrieval module**

Create `backend/app/retrieval/runtime.py` with this content:

```python
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from ..adapters.in_memory import InMemoryBM25Store
from ..adapters.neo4j import Neo4jGraphStore
from ..adapters.qdrant import QdrantVectorStore
from ..providers import EmbeddingProvider
from ..stores import BM25Document, SearchCandidate
from .pipeline import OnlineQueryPipeline, RetrievalDocument


RetrievalBackend = Literal["local", "stores"]
JsonMap = dict[str, Any]


class RuntimeRetrievalError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeRetrievalSettings:
    backend: RetrievalBackend = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "programming_chunks"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    bm25_index_path: Path = Path("data/processed/bm25_index.json")


@dataclass(frozen=True)
class RuntimeRetrieval:
    backend: RetrievalBackend
    pipeline: OnlineQueryPipeline
    candidate_sources: dict[str, str]


class JsonBM25Store:
    def __init__(self, documents: Sequence[BM25Document]) -> None:
        self._store = InMemoryBM25Store()
        self._store.index_documents(tuple(documents))

    @classmethod
    def from_path(cls, path: Path) -> JsonBM25Store:
        if not path.exists():
            raise RuntimeRetrievalError(f"BM25 index not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_documents = payload.get("documents")
        if not isinstance(raw_documents, list):
            raise RuntimeRetrievalError(f"BM25 index must contain a documents list: {path}")

        documents: list[BM25Document] = []
        for raw in raw_documents:
            if not isinstance(raw, dict):
                raise RuntimeRetrievalError(f"BM25 document must be an object in: {path}")
            document_id = str(raw["id"])
            text = str(raw.get("text") or "")
            document_payload = dict(raw.get("payload") or {})
            if "problemId" not in document_payload and raw.get("problemId") is not None:
                document_payload["problemId"] = str(raw["problemId"])
            documents.append(
                BM25Document(
                    id=document_id,
                    text=text,
                    payload=document_payload,
                )
            )
        return cls(tuple(documents))

    def index_documents(self, documents: Sequence[BM25Document]) -> None:
        self._store.index_documents(documents)

    def search(self, query: str, *, top_k: int) -> tuple[SearchCandidate, ...]:
        return self._store.search(query, top_k=top_k)


def load_runtime_retrieval_settings(
    environ: Mapping[str, str] | None = None,
) -> RuntimeRetrievalSettings:
    values = os.environ if environ is None else environ
    backend = values.get("RETRIEVAL_BACKEND", "local").strip().lower()
    if backend not in {"local", "stores"}:
        raise RuntimeRetrievalError(f"unsupported RETRIEVAL_BACKEND: {backend}")

    bm25_path = _resolve_repo_path(values.get("BM25_INDEX_PATH", "data/processed/bm25_index.json"))
    return RuntimeRetrievalSettings(
        backend=backend,  # type: ignore[arg-type]
        qdrant_url=values.get("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=values.get("QDRANT_COLLECTION", "programming_chunks"),
        neo4j_uri=values.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=values.get("NEO4J_USER", "neo4j"),
        neo4j_password=values.get("NEO4J_PASSWORD", "password"),
        bm25_index_path=bm25_path,
    )


def build_runtime_retrieval(
    settings: RuntimeRetrievalSettings | None = None,
    *,
    documents: Sequence[RetrievalDocument] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> RuntimeRetrieval:
    resolved = settings or load_runtime_retrieval_settings()
    if resolved.backend == "local":
        return RuntimeRetrieval(
            backend="local",
            pipeline=OnlineQueryPipeline(
                documents=documents,
                embedding_provider=embedding_provider,
            ),
            candidate_sources={"vector": "local", "graph": "local", "bm25": "local"},
        )

    # Qdrant and Neo4j constructors do not intentionally perform health checks here.
    # Connection problems may surface when the first query executes.
    vector_store = QdrantVectorStore(
        url=resolved.qdrant_url,
        collection_name=resolved.qdrant_collection,
    )
    graph_store = Neo4jGraphStore(
        uri=resolved.neo4j_uri,
        user=resolved.neo4j_user,
        password=resolved.neo4j_password,
    )
    bm25_store = JsonBM25Store.from_path(resolved.bm25_index_path)
    return RuntimeRetrieval(
        backend="stores",
        pipeline=OnlineQueryPipeline(
            documents=documents,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            graph_store=graph_store,
            bm25_store=bm25_store,
        ),
        candidate_sources={"vector": "qdrant", "graph": "neo4j", "bm25": "bm25_index"},
    )


def add_runtime_debug_trace(
    trace: JsonMap,
    candidate_sources: Mapping[str, str],
) -> JsonMap:
    labeled = dict(trace)
    source_labels = dict(candidate_sources)
    labeled["candidateSources"] = source_labels
    for key, source_key in (
        ("vectorCandidates", "vector"),
        ("graphCandidates", "graph"),
        ("bm25Candidates", "bm25"),
    ):
        labeled[key] = [
            {**dict(candidate), "candidateSource": source_labels[source_key]}
            for candidate in labeled.get(key, [])
        ]
    for key in ("fusionScores", "rerankerScores"):
        labeled[key] = [
            {**dict(candidate), "candidateSource": "hybrid"}
            for candidate in labeled.get(key, [])
        ]
    return labeled


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
```

- [ ] **Step 2: Run runtime retrieval tests**

Run:

```powershell
python -m pytest tests/backend/test_runtime_retrieval.py -v
```

Expected: PASS.

- [ ] **Step 3: Run existing online retrieval tests**

Run:

```powershell
python -m pytest tests/backend/test_online_retrieval_pipeline.py -v
```

Expected: PASS, proving this task did not change existing pipeline behavior or replace the runtime document candidate set.

- [ ] **Step 4: Confirm this task did not add a processed problem document loader**

Run:

```powershell
rg -n "processed/problems|problems.json|load_processed" backend/app/retrieval backend/app/main.py
```

Expected: no output.

---

### Task 3: Wire FastAPI Runtime Backend And Debug Fields

**Files:**
- Modify: `tests/backend/test_analysis.py`
- Modify: `backend/app/main.py`
- Test: `tests/backend/test_analysis.py`

- [ ] **Step 1: Extend API debug test expectations**

Replace `test_analysis_debug_mode_includes_context_preview` in `tests/backend/test_analysis.py` with this test:

```python
def test_analysis_debug_mode_includes_context_preview_and_retrieval_backend():
    client = TestClient(app)

    response = client.post(
        "/api/analysis?debug=true",
        json={"input": "unweighted graph shortest path BFS"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrievalBackend"] == "local"
    assert "contextPreview" in payload
    assert "Query Understanding" in payload["contextPreview"]
    assert payload["retrievalTrace"]["candidateSources"] == {
        "vector": "local",
        "graph": "local",
        "bm25": "local",
    }
    assert payload["retrievalTrace"]["vectorCandidates"][0]["candidateSource"] == "local"
    assert payload["retrievalTrace"]["bm25Candidates"][0]["candidateSource"] == "local"
```

- [ ] **Step 2: Preserve non-debug response shape**

In `test_analysis_response_includes_trace_and_evidence_without_context_by_default`, add this assertion after `assert "contextPreview" not in payload`:

```python
    assert "retrievalBackend" not in payload
    assert "candidateSources" not in payload["retrievalTrace"]
```

- [ ] **Step 3: Run the updated API tests and confirm failure**

Run:

```powershell
python -m pytest tests/backend/test_analysis.py::test_analysis_debug_mode_includes_context_preview_and_retrieval_backend tests/backend/test_analysis.py::test_analysis_response_includes_trace_and_evidence_without_context_by_default -v
```

Expected: FAIL because `retrievalBackend` and debug candidate labels are not wired yet.

- [ ] **Step 4: Import runtime helpers in `backend/app/main.py`**

Add this import under the existing retrieval pipeline import:

```python
from .retrieval.runtime import RuntimeRetrieval, add_runtime_debug_trace, build_runtime_retrieval
```

- [ ] **Step 5: Add `retrievalBackend` to `AnalysisResponse`**

Add this optional field to `AnalysisResponse` after `retrievalConfig`:

```python
    retrievalBackend: Literal["local", "stores"] | None = None
```

- [ ] **Step 6: Initialize runtime retrieval at startup and lazily for tests**

Use `app.on_event("startup")` in this phase. Do not refactor to FastAPI lifespan as part of this task.

Add these functions after the CORS middleware block in `backend/app/main.py`:

```python
@app.on_event("startup")
def configure_runtime_retrieval() -> None:
    app.state.runtime_retrieval = build_runtime_retrieval()


def _runtime_retrieval() -> RuntimeRetrieval:
    runtime = getattr(app.state, "runtime_retrieval", None)
    if runtime is None:
        runtime = build_runtime_retrieval()
        app.state.runtime_retrieval = runtime
    return runtime
```

- [ ] **Step 7: Use the configured runtime pipeline in `analysis`**

Replace this line in `analysis`:

```python
    pipeline_result = OnlineQueryPipeline().run(text, top_k=5)
```

with:

```python
    runtime_retrieval = _runtime_retrieval()
    pipeline_result = runtime_retrieval.pipeline.run(text, top_k=5)
    retrieval_trace = pipeline_result.trace.to_mapping()
    if debug:
        retrieval_trace = add_runtime_debug_trace(
            retrieval_trace,
            runtime_retrieval.candidate_sources,
        )
```

Then replace the response fields:

```python
        retrievalTrace=pipeline_result.trace.to_mapping(),
        evidenceBundle=evidence_bundle.to_mapping(),
        contextPreview=context_preview if debug else None,
```

with:

```python
        retrievalBackend=runtime_retrieval.backend if debug else None,
        retrievalTrace=retrieval_trace,
        evidenceBundle=evidence_bundle.to_mapping(),
        contextPreview=context_preview if debug else None,
```

- [ ] **Step 8: Remove unused direct `OnlineQueryPipeline` import if ruff reports it**

If `ruff` reports `OnlineQueryPipeline` as unused, change the import:

```python
from .retrieval.pipeline import ContextBuilder, EvidenceBuilder, OnlineQueryPipeline
```

to:

```python
from .retrieval.pipeline import ContextBuilder, EvidenceBuilder
```

- [ ] **Step 9: Run API tests**

Run:

```powershell
python -m pytest tests/backend/test_analysis.py -v
```

Expected: PASS.

- [ ] **Step 10: Run backend tests that import FastAPI app**

Run:

```powershell
python -m pytest tests/backend/test_api.py tests/backend/test_analysis.py -v
```

Expected: PASS.

---

### Task 4: Update Environment Example

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Replace the retrieval/store block**

In `.env.example`, replace the existing Neo4j/Qdrant block with:

```dotenv
# Runtime retrieval backend. local runs without Docker; stores uses Qdrant,
# Neo4j, and data/processed/bm25_index.json.
RETRIEVAL_BACKEND=local

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=programming_chunks

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

BM25_INDEX_PATH=data/processed/bm25_index.json
```

- [ ] **Step 2: Verify `.env.example` contains every required variable**

Run:

```powershell
rg -n "RETRIEVAL_BACKEND|QDRANT_URL|QDRANT_COLLECTION|NEO4J_URI|NEO4J_USER|NEO4J_PASSWORD|BM25_INDEX_PATH" .env.example
```

Expected: all seven variables are printed.

- [ ] **Step 3: Run runtime settings tests after env docs change**

Run:

```powershell
python -m pytest tests/backend/test_runtime_retrieval.py::test_load_runtime_retrieval_settings_defaults_to_local tests/backend/test_runtime_retrieval.py::test_load_runtime_retrieval_settings_rejects_unknown_backend -v
```

Expected: PASS.

---

### Task 5: Update Quick Start For Store-Backed Demo

**Files:**
- Modify: `scripts/quick-start.ps1`
- Test: `scripts/quick-start.ps1 -Check`

- [ ] **Step 1: Replace the quick-start script**

Replace `scripts/quick-start.ps1` with this ASCII-only script:

```powershell
[CmdletBinding()]
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [switch]$SkipInstall,
    [switch]$Check,
    [switch]$Stores,
    [switch]$SkipDocker,
    [switch]$SkipIngestion
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$FrontendDir = Join-Path $RepoRoot "frontend"
$RawDataDir = Join-Path $RepoRoot "data\raw"
$ProcessedDataDir = Join-Path $RepoRoot "data\processed"
$RetrievalBackend = if ($Stores) { "stores" } else { "local" }
$QdrantUrl = "http://localhost:6333"
$QdrantCollection = "programming_chunks"
$Neo4jUri = "bolt://localhost:7687"
$Neo4jUser = "neo4j"
$Neo4jPassword = "password"
$Bm25IndexPath = Join-Path $ProcessedDataDir "bm25_index.json"

function Write-Step {
    param([string]$Message)
    Write-Host "[quick-start] $Message"
}

function Resolve-RequiredCommand {
    param(
        [string]$Name,
        [string[]]$FallbackNames = @()
    )

    $names = @($Name) + $FallbackNames
    foreach ($candidate in $names) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    throw "Required command not found: $Name"
}

function Wait-HttpEndpoint {
    param(
        [string]$Url,
        [int]$Attempts = 30
    )

    for ($index = 1; $index -le $Attempts; $index++) {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 | Out-Null
            return
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }

    throw "Timed out waiting for $Url"
}

$PythonCommand = Resolve-RequiredCommand -Name "python"
$NpmCommand = Resolve-RequiredCommand -Name "npm.cmd" -FallbackNames @("npm")
$DockerCommand = $null
if ($Stores) {
    $DockerCommand = Resolve-RequiredCommand -Name "docker"
}

if (-not (Test-Path (Join-Path $RepoRoot "backend\app\main.py"))) {
    throw "Backend entry not found: backend\app\main.py"
}

if (-not (Test-Path (Join-Path $FrontendDir "package.json"))) {
    throw "Frontend package.json not found"
}

if ($Check) {
    Write-Step "Workspace: $RepoRoot"
    Write-Step "Python: $PythonCommand"
    Write-Step "npm: $NpmCommand"
    Write-Step "Retrieval backend: $RetrievalBackend"
    Write-Step "Backend URL: http://127.0.0.1:$BackendPort"
    Write-Step "Frontend URL: http://127.0.0.1:$FrontendPort"
    if ($Stores) {
        Write-Step "Docker: $DockerCommand"
        Write-Step "Qdrant URL: $QdrantUrl"
        Write-Step "Qdrant collection: $QdrantCollection"
        Write-Step "Neo4j URI: $Neo4jUri"
        Write-Step "BM25 index: $Bm25IndexPath"
    }
    Write-Step "Check complete. No services started."
    exit 0
}

if (-not $SkipInstall -and -not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
    Write-Step "Installing frontend dependencies"
    Push-Location $FrontendDir
    try {
        & $NpmCommand install
    }
    finally {
        Pop-Location
    }
}

if ($Stores) {
    if (-not $SkipDocker) {
        Write-Step "Starting Neo4j and Qdrant with Docker Compose"
        Push-Location $RepoRoot
        try {
            & $DockerCommand compose up -d neo4j qdrant
        }
        finally {
            Pop-Location
        }
        Write-Step "Waiting for Qdrant"
        Wait-HttpEndpoint -Url $QdrantUrl
        Write-Step "Waiting for Neo4j browser endpoint"
        Wait-HttpEndpoint -Url "http://localhost:7474"
    }

    if (-not $SkipIngestion) {
        Write-Step "Running ingestion into Qdrant, Neo4j, and BM25 index"
        Push-Location $RepoRoot
        try {
            & $PythonCommand -m backend.app.ingestion build `
                --input $RawDataDir `
                --processed $ProcessedDataDir `
                --target all
        }
        finally {
            Pop-Location
        }
    }
}

Write-Step "Starting backend and frontend. Press Ctrl+C to stop."
Write-Step "Retrieval backend: $RetrievalBackend"
Write-Step "Backend: http://127.0.0.1:$BackendPort"
Write-Step "Frontend: http://127.0.0.1:$FrontendPort"

$backendJob = Start-Job -Name "knowledgegraph-rag-backend" -ScriptBlock {
    param(
        $Root,
        $Port,
        $Python,
        $Backend,
        $QdrantUrlValue,
        $QdrantCollectionValue,
        $Neo4jUriValue,
        $Neo4jUserValue,
        $Neo4jPasswordValue,
        $Bm25IndexValue
    )
    Set-Location $Root
    $env:PYTHONPATH = $Root
    $env:RETRIEVAL_BACKEND = $Backend
    $env:QDRANT_URL = $QdrantUrlValue
    $env:QDRANT_COLLECTION = $QdrantCollectionValue
    $env:NEO4J_URI = $Neo4jUriValue
    $env:NEO4J_USER = $Neo4jUserValue
    $env:NEO4J_PASSWORD = $Neo4jPasswordValue
    $env:BM25_INDEX_PATH = $Bm25IndexValue
    & $Python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port $Port
} -ArgumentList $RepoRoot,
    $BackendPort,
    $PythonCommand,
    $RetrievalBackend,
    $QdrantUrl,
    $QdrantCollection,
    $Neo4jUri,
    $Neo4jUser,
    $Neo4jPassword,
    $Bm25IndexPath

$frontendJob = Start-Job -Name "knowledgegraph-rag-frontend" -ScriptBlock {
    param($Frontend, $Port, $Npm)
    Set-Location $Frontend
    & $Npm run dev -- --host 127.0.0.1 --port $Port
} -ArgumentList $FrontendDir, $FrontendPort, $NpmCommand

try {
    while ($true) {
        foreach ($job in @($backendJob, $frontendJob)) {
            Receive-Job -Job $job -Keep | ForEach-Object { Write-Host $_ }
            if ($job.State -in @("Failed", "Stopped", "Completed")) {
                throw "Service stopped: $($job.Name) ($($job.State))"
            }
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Step "Stopping services"
    Stop-Job -Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $backendJob, $frontendJob -Force -ErrorAction SilentlyContinue
}
```

- [ ] **Step 2: Run quick-start preflight in local mode**

Run:

```powershell
.\scripts\quick-start.ps1 -Check
```

Expected: PASS and prints `Retrieval backend: local`.

- [ ] **Step 3: Run quick-start preflight in store mode**

Run:

```powershell
.\scripts\quick-start.ps1 -Check -Stores
```

Expected: PASS if Docker is installed, prints `Retrieval backend: stores`, Qdrant URL, Neo4j URI, and BM25 index path. If Docker is not installed, the failure should be `Required command not found: docker`.

---

### Task 6: Add Optional Docker Integration Smoke Test

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/backend/test_store_runtime_integration.py`

- [ ] **Step 1: Register the Docker marker**

In `pyproject.toml`, update `[tool.pytest.ini_options]` to:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = [
  "docker: requires local Docker services and seeded stores",
]
```

- [ ] **Step 2: Add the skipped-by-default integration test**

Create `tests/backend/test_store_runtime_integration.py`:

```python
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
```

- [ ] **Step 3: Confirm default backend suite does not require Docker**

Run:

```powershell
python -m pytest tests/backend/test_store_runtime_integration.py -v
```

Expected: the test is collected and skipped unless `RUN_DOCKER_TESTS=1`.

---

### Task 7: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/api.md`
- Modify: `scripts/README.md`

- [ ] **Step 1: Add runtime backend section to `README.md`**

Add this section after the ingestion artifact list:

````markdown
## Runtime Retrieval Backend

FastAPI supports two retrieval backends:

```text
RETRIEVAL_BACKEND=local
RETRIEVAL_BACKEND=stores
```

`local` is the default. It uses the local fallback documents and does not require Docker.

`stores` creates a Qdrant vector store, a Neo4j graph store, and a BM25 store loaded from `data/processed/bm25_index.json`. Use the store-backed quick start to seed and run the demo:

```powershell
.\scripts\quick-start.ps1 -Stores
```

Debug the active backend and retrieval provenance with:

```powershell
curl.exe -X POST "http://localhost:8000/api/analysis?debug=true" `
  -H "Content-Type: application/json" `
  -d "{\"input\":\"unweighted graph shortest path BFS\"}"
```

The debug response includes:

```text
retrievalBackend
retrievalTrace.candidateSources.vector = qdrant
retrievalTrace.candidateSources.graph = neo4j
retrievalTrace.candidateSources.bm25 = bm25_index
```

This phase keeps the existing runtime documents as the graph candidate set. It does not load `data/processed/problems.json` or require store hits to include full answers or solution hints.
````

- [ ] **Step 2: Replace the deferred runtime note in `docs/architecture.md`**

Replace the paragraph that says FastAPI runtime store mode is not part of the phase with:

````markdown
## Runtime Backend Selection

FastAPI reads `RETRIEVAL_BACKEND` at startup:

- `local`: builds the default `OnlineQueryPipeline()` and keeps the local fallback behavior.
- `stores`: builds `QdrantVectorStore`, `Neo4jGraphStore`, and `JsonBM25Store`, then injects them into `OnlineQueryPipeline`.

This is runtime wiring, not a full dataset source replacement. `OnlineQueryPipeline` still uses the existing runtime documents as the graph candidate set. Loading documents from `data/processed/problems.json` belongs in a separate follow-up phase.

`JsonBM25Store` reads `data/processed/bm25_index.json` immediately. Qdrant and Neo4j constructors do not intentionally run health checks here, so connection failures may surface on the first query unless a future task adds explicit health checks.

The API keeps the existing logical retrieval lanes: vector, graph, BM25, fusion, and rerank. In debug mode, `retrievalTrace.candidateSources` identifies the physical backend for each lane:

```json
{
  "vector": "qdrant",
  "graph": "neo4j",
  "bm25": "bm25_index"
}
```

Graph store paths keep two shapes:

- `nodes` / `relations`: stable display summary, `input -> linked entity -> problem`.
- `storePath.nodes` / `storePath.relations`: raw nodes and relations returned by Neo4j.
````

- [ ] **Step 3: Update `docs/api.md` analysis debug contract**

In the Analysis Response section, add `retrievalBackend` to the debug-only fields:

```text
retrievalTrace
evidenceBundle
contextPreview
retrievalBackend
```

Replace the Retrieval Trace example with:

```json
{
  "queryUnderstanding": {
    "intent": "problem_search",
    "inputKind": "problem",
    "keywords": ["unweighted", "graph", "shortest", "path", "bfs"]
  },
  "entityLinking": [],
  "candidateSources": {
    "vector": "qdrant",
    "graph": "neo4j",
    "bm25": "bm25_index"
  },
  "vectorCandidates": [
    {
      "id": "leetcode-994",
      "source": "vector",
      "candidateSource": "qdrant"
    }
  ],
  "graphCandidates": [
    {
      "id": "leetcode-994",
      "source": "graph",
      "candidateSource": "neo4j"
    }
  ],
  "bm25Candidates": [
    {
      "id": "leetcode-994",
      "source": "bm25",
      "candidateSource": "bm25_index"
    }
  ],
  "fusionScores": [],
  "rerankerScores": []
}
```

Add this note below the example:

```markdown
`candidateSource` is only added when `debug=true`. Non-debug responses keep the existing `retrievalTrace` shape and omit `retrievalBackend`.

Store-backed vector and BM25 candidates are chunk-level hits and may omit full answers or solution hints. The graph lane still uses the runtime document set as candidates in this phase.
```

- [ ] **Step 4: Update `scripts/README.md`**

Replace the Quick Start section with:

````markdown
# Scripts

## Quick Start

Run the local fallback demo without Docker:

```powershell
.\scripts\quick-start.ps1
```

Run the store-backed demo with Neo4j, Qdrant, ingestion, FastAPI, and Vite:

```powershell
.\scripts\quick-start.ps1 -Stores
```

Check prerequisites and paths without starting services:

```powershell
.\scripts\quick-start.ps1 -Check
.\scripts\quick-start.ps1 -Check -Stores
```

The script starts:

- FastAPI at `http://127.0.0.1:8000`
- Vite at `http://127.0.0.1:5173`
- Neo4j at `http://localhost:7474` and `bolt://localhost:7687` when `-Stores` is used
- Qdrant at `http://localhost:6333` when `-Stores` is used

Use `-SkipDocker` after Docker services are already running. Use `-SkipIngestion` after `data/processed/bm25_index.json`, Qdrant, and Neo4j are already seeded.
````

- [ ] **Step 5: Verify docs keep this phase scoped to runtime wiring**

Run:

```powershell
rg -n "runtime wiring|data/processed/problems.json|first query|solution hints|candidate set" README.md docs/architecture.md docs/api.md scripts/README.md
```

Expected: output includes the runtime-wiring scope, the explicit `data/processed/problems.json` deferral, and first-query connection wording.

---

### Task 8: Full Verification

**Files:**
- Verify: all changed files

- [ ] **Step 1: Run ruff**

Run:

```powershell
python -m ruff check .
```

Expected: PASS.

- [ ] **Step 2: Run backend tests**

Run:

```powershell
python -m pytest tests/backend
```

Expected: PASS. The Docker integration smoke test is skipped unless `RUN_DOCKER_TESTS=1`.

- [ ] **Step 3: Run frontend build**

Run:

```powershell
cd frontend
npm.cmd run build
```

Expected: PASS.

- [ ] **Step 4: Run quick-start checks**

Run from the repo root:

```powershell
.\scripts\quick-start.ps1 -Check
.\scripts\quick-start.ps1 -Check -Stores
```

Expected:
- Local check prints `Retrieval backend: local`.
- Store check prints `Retrieval backend: stores` and the configured Qdrant, Neo4j, and BM25 paths. If Docker is not installed on the machine, record the exact `Required command not found: docker` message instead of treating it as a code failure.

- [ ] **Step 5: Optional manual store-backed runtime smoke**

Run:

```powershell
.\scripts\quick-start.ps1 -Stores
```

Then in another terminal:

```powershell
curl.exe -X POST "http://localhost:8000/api/analysis?debug=true" `
  -H "Content-Type: application/json" `
  -d "{\"input\":\"unweighted graph shortest path BFS\"}"
```

Expected response contains:

```json
{
  "retrievalBackend": "stores",
  "retrievalTrace": {
    "candidateSources": {
      "vector": "qdrant",
      "graph": "neo4j",
      "bm25": "bm25_index"
    }
  }
}
```

- [ ] **Step 6: Inspect changed files**

Run:

```powershell
git status --short
git diff -- backend/app/main.py backend/app/retrieval/runtime.py tests/backend/test_runtime_retrieval.py tests/backend/test_analysis.py .env.example scripts/quick-start.ps1 pyproject.toml README.md docs/architecture.md docs/api.md scripts/README.md
```

Expected: changes are limited to runtime wiring, tests, quick-start, env example, pytest marker, and docs.

---

## Self-Review

- Spec coverage:
  - `RETRIEVAL_BACKEND=local` and `RETRIEVAL_BACKEND=stores`: Tasks 1, 2, 4.
  - Local mode keeps fallback and does not require Docker: Tasks 1, 2, 5, 8.
  - Store mode creates Qdrant, Neo4j, and BM25Store from `data/processed/bm25_index.json`: Task 2.
  - Store instances are injected into `OnlineQueryPipeline`: Task 2.
  - `.env.example` includes all requested variables: Task 4.
  - `scripts/quick-start.ps1` starts Neo4j/Qdrant, runs ingestion, starts FastAPI, and starts frontend when `-Stores` is used: Task 5.
  - Debug response includes `retrievalBackend`: Task 3.
  - Debug trace shows `vector: qdrant`, `graph: neo4j`, and `bm25: bm25_index`: Tasks 2, 3, 7.
  - Tests cover local mode, store injection with fakes, and Docker test marker behavior: Tasks 1, 2, 6.
  - README, architecture, API, and scripts docs are updated: Task 7.
  - Required verification commands are included: Task 8.
- Red-flag scan:
  - No task relies on undefined functions or unnamed files.
  - Code-changing steps include exact code blocks.
  - Docker-backed testing is explicitly skipped by default and does not become a CI dependency.
- Type consistency:
  - `RuntimeRetrievalSettings.backend` uses `Literal["local", "stores"]`.
  - `RuntimeRetrieval.pipeline` remains an `OnlineQueryPipeline`.
  - `JsonBM25Store` implements the existing `BM25Store` protocol methods.
  - Debug source labels are strings and do not replace `RetrievalCandidate.source`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-22-end-to-end-store-backed-demo.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using `superpowers:executing-plans`, with checkpoints after each task group.

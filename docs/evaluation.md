# Evaluation Plan

The evaluation should test whether hybrid retrieval gives better and more
explainable recommendations than simpler baselines on a fixed dataset.

## Baselines

- Vector-only: rank candidates by semantic similarity.
- Graph-only: rank candidates by graph evidence from known problem/concept
  relations.
- Hybrid: combine vector similarity, graph path support, and concept overlap.
- No-LLM: return evidence bundles without natural-language generation.

## Metrics

- Top-1, Top-3, and Top-5 algorithm hit rate.
- Top-1, Top-3, and Top-5 data-structure hit rate.
- Pattern hit rate.
- Exclusion correctness for cases where a tempting algorithm should be ruled out.
- Human review of evidence paths for a 30-problem sample.

## Controls

- Use the same embedding model and candidate pool for vector-only and hybrid.
- Ensure `mode=vector`, `mode=graph`, and `mode=hybrid` each restrict final
  candidates to the selected lane or fusion path before comparing metrics.
- Keep service-level store-backed retrieval tests comparable with the local
  fallback fixtures, so `VectorStore`, `BM25Store`, and `GraphStore` injection
  can be compared without changing expected answers.
- Run runtime stores checks with
  `PROCESSED_PROBLEMS_PATH=data/processed/problems.json`, so the live API path
  covers processed runtime documents and enriched store payloads.
- Freeze the test set before tuning ranking weights.
- Keep LLM output out of scoring unless evaluating explanation readability.
- Keep Query Understanding rule-based for this phase; real LLM-backed query
  understanding is a later architecture option, not part of the current
  retrieval refactor.
- Record errors by category: missing graph edge, wrong label, weak embedding
  match, ambiguous problem statement, or LLM wording issue.

## Minimum Acceptance for v1

- The system can run with no dataset and no external services using in-memory
  repositories.
- A later dataset import can populate the same `Problem`, `Concept`, and
  relationship contracts.
- Retrieval returns evidence paths and scores that can be inspected without
  trusting LLM prose.
- The evaluation harness can compare vector-only, graph-only, and hybrid modes
  on the same fixture format.

## Post-QA Regression Cases

| Case | Expected contract |
|---|---|
| Backend stopped | The API returns an explicit error; it must not return a mock success response. |
| Exact UVA query | `UVA-10653 - Bombs! NO they are Mines!!`, bare `10653`, and a partial UVA title resolve to canonical `matchedProblem.id=uva-10653`; chunk evidence covers statement, answer, and hints; graph evidence is canonical, operation-labeled, layered, typed, weighted, and path-scored. |
| Equivalent BFS code | Equivalent C++ and Python BFS inputs link `BFS`, `Queue`, and `Visited Array` through `code_feature:*` IDs. C++ candidate chunk provenance is complete or identifies missing sources. |
| Chinese BFS text | A Chinese BFS problem description remains supported and returns the corresponding concepts and evidence. |
| Unrelated text | Dinner, weather, or MRT text returns `status=unsupported` with no BFS concepts, recommendations, or graph evidence. |
| Whitespace input | Whitespace-only input is rejected as invalid input. |
| Oversized input | A 96,000-character input returns HTTP `413` with `input_too_large`. |
| Score labels | Every visible candidate and graph-path score has stage metadata; UI and evaluators do not compare values across scoring stages without that metadata. |
| Identifier joins | Response records use canonical `id` and retain source-local `sourceId` where available. |

Run the matrix with direct `TestClient` or contract-level checks before relying
on browser results. Run the stores-mode configuration check without starting
external Qdrant or Neo4j; before version alignment, debug diagnostics must make
any store compatibility warning visible.

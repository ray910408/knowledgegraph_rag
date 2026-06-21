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
- Freeze the test set before tuning ranking weights.
- Keep LLM output out of scoring unless evaluating explanation readability.
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

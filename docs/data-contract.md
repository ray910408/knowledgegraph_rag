# Data Contract

## Raw Problem Schema

## Identifier Contract

`id` is the canonical cross-system identifier, for example `leetcode-1091` or
`uva-10653`. `sourceId` is the source-local identifier, for example `1091` or
`10653`, when that source provides one. Join records by canonical `id`; retain
`sourceId` for display and source-specific references.

`data/raw/*.json` 可放單一 object、array，或 `{ "problems": [...] }`。

Repo 內建 `data/raw/programming_problems.json` 作為可執行的 seed fixture。這個檔案必須保持 UTF-8 readable zh-Hant，中文內容不要寫成 `\uXXXX` escape。

必填欄位：

```text
id
source
sourceId
title
problemType
statement
answer
solutionHints
concepts
tags
metadata
```

可選欄位：

```text
difficulty
constraints
examples
editorial
```

範例：

```json
{
  "id": "leetcode-994",
  "source": "LeetCode",
  "sourceId": "994",
  "title": "Rotting Oranges",
  "problemType": "Graph Traversal",
  "statement": "Multi-source BFS on a grid.",
  "answer": "Use BFS from all initially rotten oranges.",
  "solutionHints": ["Push all sources first."],
  "concepts": ["BFS", "Queue"],
  "tags": ["matrix", "graph"],
  "metadata": { "url": "https://leetcode.com/problems/rotting-oranges/" },
  "difficulty": "Medium",
  "constraints": ["1 <= m, n <= 10"],
  "examples": [{ "input": "grid", "output": "4" }],
  "editorial": "This is a multi-source BFS problem."
}
```

## Processed Artifacts

### `problems.json`

清理與標準化後的 raw problem。去重 key 為 `source + sourceId`。

### `chunks.json`

欄位：

```text
id
problemId
kind
text
index
answer
solutionHints
difficulty
constraints
examples
editorial
source
sourceId
title
problemType
concepts
metadata
```

`kind` 可為：

```text
statement
answer
hint
editorial
```

### `entities.json`

欄位：

```text
id
name
type
aliases
problemIds
metadata
```

常見 `type`：

```text
problem
algorithm
data_structure
pattern
concept
```

### `relations.json`

欄位：

```text
id
sourceId
targetId
type
weight
evidence
metadata
```

常見 relation：

```text
REQUIRES
HAS_PATTERN
```

### `bm25_index.json`

本地 BM25 artifact，包含 documents、tokens 與 chunk payload。

BM25 document payloads include the enriched evidence fields used at runtime:

```text
answer
solutionHints
difficulty
constraints
examples
editorial
source
sourceId
title
problemType
concepts
```

This is the raw BM25 artifact/store payload shape. Normalized retrieval
candidate `payload` maps `source` to `documentSource`; `storePayload` retains
`source`.

### `qdrant_vectors.json`

欄位：

```text
embeddingModel
records
```

每筆 record：

```text
id
vector
payload
```

Qdrant payload includes chunk identity plus enriched evidence fields:

```text
answer
solutionHints
difficulty
constraints
examples
editorial
source
sourceId
title
problemType
concepts
```

This is the raw Qdrant artifact/store payload shape. Normalized retrieval
candidate `payload` maps `source` to `documentSource`; `storePayload` retains
`source`.

### `neo4j_graph.json`

欄位：

```text
entities
relations
```

這份 artifact 可作為 Neo4j import/debug 的中介格式。

## Retrieval Trace Contract

`retrievalTrace` 包含：

```text
queryUnderstanding
entityLinking
vectorCandidates
graphCandidates
bm25Candidates
fusionScores
rerankerScores
candidateSources
providerSources
compatibilityWarnings
matchedProblem
```

`matchedProblem` appears when exact problem ID, source ID, or title matching
pins a canonical problem separately from the reranked similar-problem list.

Store-backed candidates can include store metadata in each candidate payload:

```text
storeCandidateId
storePayload
```

`candidateSources` and `providerSources` are optional diagnostics that describe
the active physical retrievers and model/provider wiring. When present,
`compatibilityWarnings` carries bounded adapter warnings surfaced by debug mode.

Scores are stage-specific. Candidate and graph-path scores include `scoreMeta`;
consumers must inspect it before comparing values. In particular, BM25, vector,
graph path, fusion, and reranker scores are not relevance-comparable merely
because they are numeric.

Candidate provenance can include:

```text
chunkEvidence.available
chunkEvidence.complete
chunkEvidence.missingSources
chunkEvidence.unavailableReason
```

`complete=true` means all contributing retrieval sources supplied usable chunk
provenance. `missingSources` lists the contributing sources whose provenance is
absent or incomplete; it is empty only when no source is missing. Consumers
should not imply complete source evidence when `complete=false`.

Vector and BM25 normalized candidate `payload` values can include enriched
evidence fields such as `answer`, `solutionHints`, `difficulty`, `constraints`,
`examples`, `editorial`, `documentSource`, `sourceId`, `title`, `problemType`,
and `concepts`. Raw `storePayload` values retain the processed/store field
name `source` instead of `documentSource`.

Store-backed graph paths preserve both the stable summary and the raw store path:

```json
{
  "nodes": [
    { "id": "uva-10653", "label": "Bombs! NO they are Mines!!", "layer": "problem" },
    { "id": "source:uva:10653", "label": "UVa 10653", "layer": "source" },
    { "id": "concept:bfs", "label": "BFS", "layer": "concept" }
  ],
  "relations": [
    {
      "source": "uva-10653",
      "target": "source:uva:10653",
      "type": "EXPANDED_FROM_EXACT_MATCH",
      "weight": 1.0
    },
    {
      "source": "source:uva:10653",
      "target": "concept:bfs",
      "type": "MENTIONS_CONCEPT",
      "weight": 1.0
    }
  ],
  "score": 0.85,
  "rationale": "Inferred from document concepts for the exact matched problem.",
  "pathSource": "inferred",
  "graphPathOperation": "exact_expansion",
  "pathScoring": {
    "strategy": "weighted_layered_path_v1",
    "score": 0.85
  },
  "scoreMeta": {
    "stage": "graph_path",
    "displayLabel": "Graph path confidence",
    "comparableAcrossStages": false
  },
  "storePath": {
    "nodes": ["uva-10653", "concept:bfs"],
    "relations": ["REQUIRES"]
  }
}
```

## Analysis Response Contract

Top-level analysis responses always include:

```text
queryId
status
abstentionReason
usedMockData
inputKind
problemType
requiredConcepts
matchedProblem
similarProblems
similarityReason
solvingHints
commonMistakes
evidencePaths
retrievalConfig
retrievalTrace
evidenceBundle
```

`contextPreview` and `retrievalBackend` are debug-only additions. `status`
distinguishes successful analysis from intentional abstention, and
`matchedProblem` carries exact canonical problem hits separately from
`similarProblems`.

## Evidence Bundle Contract

`evidenceBundle` 包含：

```text
similarProblems
graphPaths
algorithmEvidence
dataStructureEvidence
techniqueEvidence
patternEvidence
commonMistakes
matchedProblem
```

`similarProblems` is built from the final selected retrieval mode candidate
set. Exact matched problems are represented separately as `matchedProblem` and
are excluded from `similarProblems`.

`graphPaths` entries can include `nodes`, `relations`, `score`, `rationale`,
`storePath`, `pathSource`, `graphPathOperation`, `pathScoring`, and `scoreMeta`.
`storePath` preserves the raw store-returned nodes and relations while
`nodes` / `relations` keep the stable display summary. `pathSource` explains
whether the path came from Neo4j or inferred fallback evidence, and
`graphPathOperation` distinguishes retrieved candidate paths from exact-match
expansion paths.

Graph paths cross a reference boundary with layered nodes (`problem`, `chunk`,
`concept`, `code_feature`, `pattern`, `source`) and typed, weighted relations.
Their deterministic scoring uses `pathScoring` with
`strategy=weighted_layered_path_v1`; graph path `score` equals
`pathScoring.score`. Relation weights express local edge confidence only and
are not interchangeable with BM25, vector, fusion, reranker, or graph-path
retrieval scores.

## Context Preview Contract

`contextPreview` 是送入 `LLMProvider` 前整理好的 prompt context。它只會在 `debug=true` 時回傳。

The context builder can use enriched candidate payload fields such as `answer`,
`solutionHints`, `difficulty`, and `constraints`, plus graph path `rationale`.
Non-debug responses omit `contextPreview`.

When `similarProblems` is empty, `ContextBuilder` omits the `相似題` section
instead of emitting an empty heading.

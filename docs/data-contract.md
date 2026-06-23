# Data Contract

## Raw Problem Schema

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
```

Store-backed candidates can include store metadata in each candidate payload:

```text
storeCandidateId
storePayload
```

Vector and BM25 normalized candidate `payload` values can include enriched
evidence fields such as `answer`, `solutionHints`, `difficulty`, `constraints`,
`examples`, `editorial`, `documentSource`, `sourceId`, `title`, `problemType`,
and `concepts`. Raw `storePayload` values retain the processed/store field
name `source` instead of `documentSource`.

Store-backed graph paths preserve both the stable summary and the raw store path:

```json
{
  "nodes": ["input", "concept:bfs", "leetcode-994"],
  "relations": ["MENTIONS", "REQUIRED_BY"],
  "rationale": "BFS is linked to the matching problem through the graph store.",
  "storePath": {
    "nodes": ["leetcode-994", "concept:bfs"],
    "relations": ["REQUIRES"]
  }
}
```

## Evidence Bundle Contract

`evidenceBundle` 包含：

```text
similarProblems
graphPaths
algorithmEvidence
dataStructureEvidence
patternEvidence
commonMistakes
```

`graphPaths` entries can include `nodes`, `relations`, `rationale`, and
`storePath`. `storePath` preserves the raw store-returned nodes and relations
while `nodes` / `relations` keep the stable display summary.

## Context Preview Contract

`contextPreview` 是送入 `LLMProvider` 前整理好的 prompt context。它只會在 `debug=true` 時回傳。

The context builder can use enriched candidate payload fields such as `answer`,
`solutionHints`, `difficulty`, and `constraints`, plus graph path `rationale`.
Non-debug responses omit `contextPreview`.

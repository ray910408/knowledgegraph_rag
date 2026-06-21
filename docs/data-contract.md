# Data Contract

## Raw Problem Schema

`data/raw/*.json` 可為單題 object、題目 array，或 `{ "problems": [...] }`。

必要欄位：

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

清理與去重後的 Raw Problem。去重 key 為 `source + sourceId`，保留第一筆。

### `chunks.json`

欄位：

```text
id
problemId
kind
text
index
concepts
metadata
```

`kind` 目前包含：

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

目前常見 relation：

```text
REQUIRES
HAS_PATTERN
```

### `bm25_index.json`

本機 BM25 artifact，包含 documents、tokens 與 chunk payload。

### `qdrant_vectors.json`

欄位：

```text
embeddingModel
records
```

每筆 record 包含：

```text
id
vector
payload
```

### `neo4j_graph.json`

欄位：

```text
entities
relations
```

此檔同時作為 Neo4j import/debug artifact。

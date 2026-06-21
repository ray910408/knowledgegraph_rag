# Data Contract

The repository does not fetch datasets in v1. User-provided files can later be
placed under `data/raw/` and converted into the graph/vector stores.

## Problem Record

```json
{
  "id": "uva-10653",
  "source": "UVa",
  "sourceId": "10653",
  "title": "Bombs! NO they are Mines!!",
  "problemType": "Graph Traversal",
  "statement": "Problem statement text",
  "answer": "Use BFS on the unweighted grid to find the minimum number of steps.",
  "solutionHints": ["Build the blocked-cell grid first.", "Run BFS from the start cell."],
  "concepts": ["BFS", "Queue", "Visited Array"],
  "difficulty": "practice",
  "tags": ["graph", "shortest-path"]
}
```

`answer` and `solutionHints` are intentionally kept in the raw dataset seed so
the system can explain why a similar problem is useful without generating full
accepted code.

## Concept Record

```json
{
  "id": "bfs",
  "kind": "Algorithm",
  "name": "Breadth First Search",
  "aliases": ["BFS"],
  "description": "Layered traversal for unweighted graphs."
}
```

## Relationship Record

```json
{
  "source_id": "cpe-0001",
  "target_id": "bfs",
  "type": "SOLVED_BY",
  "weight": 1.0,
  "rationale": "Unweighted shortest path."
}
```

## Evaluation Split

Recommended first evaluation set:

- 100 CPE 1-star to 3-star problems
- 30 LeetCode Easy problems
- Frozen labels for algorithm, data structure, pattern, and exclusion cases

Do not include evaluation labels in the retrieval prompt or LLM evidence unless
the test explicitly measures label leakage.

# API Contract

## `GET /api/health`

Returns a simple liveness response.

```json
{
  "status": "ok"
}
```

Equivalent versioned route:

- `GET /api/v1/health`

## `POST /api/analysis`

Primary endpoint for the explainable problem-solving assistant. The web app
sends camelCase fields. The backend also accepts snake_case and `statement` for
script compatibility.

Request:

```json
{
  "inputText": "Problem statement or pasted C++/Python code",
  "mode": "hybrid",
  "topK": 4
}
```

Compatible request:

```json
{
  "statement": "Given an unweighted graph, find the minimum number of steps.",
  "top_k": 4
}
```

Equivalent versioned route:

- `POST /api/v1/analysis`

Response:

```json
{
  "queryId": "analysis-problem-graph-traversal",
  "usedMockData": false,
  "inputKind": "problem",
  "problemType": "圖論遍歷（Graph Traversal）",
  "requiredConcepts": [
    {
      "id": "bfs",
      "name": "BFS",
      "kind": "algorithm",
      "description": "在無權圖中依照層次展開搜尋，適合找最短步數。"
    },
    {
      "id": "queue",
      "name": "Queue",
      "kind": "data_structure",
      "description": "維持 BFS 的先進先出順序。"
    }
  ],
  "similarProblems": [
    {
      "source": "UVa",
      "id": "10653",
      "title": "Bombs! NO they are Mines!!",
      "reason": "同樣需要在無權圖中用 BFS 找最短步數。",
      "sharedConcepts": ["BFS", "Queue", "Visited Array"],
      "answerHint": "先建圖，再從起點做 BFS。"
    }
  ],
  "similarityReason": "都需要在無權圖中用 BFS 找最短步數，並用 Queue 維持搜尋層次。",
  "solvingHints": ["先建圖或定義狀態轉移。", "從起點初始化 Queue。", "再執行 BFS。"],
  "commonMistakes": ["忘記標記 visited。", "Queue 初始化錯誤。"],
  "evidencePaths": [
    {
      "title": "無權最短路徑推理",
      "nodes": [
        { "id": "input", "label": "輸入題目", "type": "problem" },
        { "id": "bfs", "label": "BFS", "type": "algorithm" }
      ],
      "edges": [
        { "from": "input", "to": "bfs", "relation": "無權最短步數", "weight": 0.92 }
      ]
    }
  ],
  "retrievalConfig": {
    "embeddingModel": "BAAI/bge-m3",
    "rerankerModel": "BAAI/bge-reranker-v2-m3",
    "language": "zh-Hant"
  }
}
```

Accepted `inputKind` values:

- `problem`
- `cpp`
- `python`
- `unknown`

## `POST /api/recommendations`

The web app sends camelCase fields. The backend also accepts `statement` and
`top_k` aliases so later scripts can use the data-contract naming style.

Request:

```json
{
  "problemText": "Problem statement or pasted prompt",
  "mode": "hybrid",
  "topK": 5
}
```

Compatible request:

```json
{
  "problem_id": "demo-shortest-subarray",
  "statement": "Problem statement or pasted prompt",
  "mode": "graph",
  "top_k": 5
}
```

Equivalent versioned route:

- `POST /api/v1/recommendations`

Accepted aliases:

- Problem id: `problemId`, `problem_id`
- Problem text: `problemText`, `problem_text`, `statement`
- Top-k: `topK`, `top_k`

If both a problem id and explicit problem text are provided, the backend uses the
explicit text for retrieval and marks the response `queryId` with `with-text`.

Modes:

- `hybrid`
- `vector`
- `graph`

Response:

```json
{
  "queryId": "demo-hybrid-5",
  "usedMockData": false,
  "recommendations": [
    {
      "id": "sliding-window",
      "kind": "pattern",
      "title": "Sliding window",
      "score": 0.91,
      "confidence": "high",
      "summary": "Maintain a moving interval for contiguous constraints.",
      "fitSignals": ["Range sum", "Monotonic window"],
      "pitfalls": ["Requires a monotonic condition such as non-negative values."]
    }
  ],
  "evidencePaths": [
    {
      "title": "Sliding window evidence 1",
      "nodes": [
        { "id": "range-sum", "label": "Range sum", "type": "concept" },
        { "id": "sliding-window", "label": "Sliding window", "type": "pattern" }
      ],
      "edges": [
        {
          "from": "range-sum",
          "to": "sliding-window",
          "relation": "HAS_PATTERN",
          "weight": 0.5
        }
      ]
    }
  ],
  "evaluation": [
    {
      "name": "Concept recall",
      "vectorOnly": 0.72,
      "graphOnly": 0.65,
      "hybrid": 0.84,
      "note": "Demo placeholder until the frozen CPE/LeetCode set is imported."
    }
  ]
}
```

The frontend treats successful but malformed responses as unusable and falls
back to local mock data, preventing the workbench from crashing on partial API
implementations.

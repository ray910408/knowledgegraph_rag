# API Contract

## Health

```http
GET /api/health
GET /api/v1/health
```

Response：

```json
{ "status": "ok" }
```

## Analysis

```http
POST /api/analysis
POST /api/v1/analysis
```

Request body：

```json
{
  "input": "unweighted graph shortest path BFS"
}
```

相容欄位：

```text
input
problemText
statement
code
problemId
```

行為：

- 空輸入回 `400`。
- 若指定 `problemId` 且不存在，回 `404`。
- 若沒有指定 `problemId`，一般文字會走 query search。

### Analysis Response

保留既有欄位：

```text
queryId
usedMockData
inputKind
problemType
requiredConcepts
similarProblems
similarityReason
solvingHints
commonMistakes
evidencePaths
retrievalConfig
```

新增可選欄位：

```text
retrievalTrace
evidenceBundle
contextPreview
```

`contextPreview` 只在 debug mode 回傳：

```http
POST /api/analysis?debug=true
```

FastAPI runtime store mode is not part of the current API contract. The Python
retrieval pipeline supports injecting `vector_store`, `bm25_store`, and
`graph_store`, but the public API still constructs the default local fallback
pipeline. Qdrant, Neo4j, and BM25Store runtime wiring belongs to the next
End-to-End Store-Backed Demo phase.

### Retrieval Trace

```json
{
  "queryUnderstanding": {
    "intent": "problem_search",
    "inputKind": "problem",
    "keywords": ["unweighted", "graph", "shortest", "path", "bfs"]
  },
  "entityLinking": [],
  "vectorCandidates": [],
  "graphCandidates": [],
  "bm25Candidates": [],
  "fusionScores": [],
  "rerankerScores": []
}
```

Store-backed graph paths use the same display summary as local graph paths:
`input -> linked entity -> problem`. When the graph store returns a raw path, the
raw `nodes` and `relations` are preserved under `storePath`.

### Evidence Bundle

```json
{
  "similarProblems": [],
  "graphPaths": [],
  "algorithmEvidence": ["BFS"],
  "dataStructureEvidence": ["Queue"],
  "patternEvidence": ["Graph Traversal"],
  "commonMistakes": []
}
```

### Example

```powershell
curl.exe -X POST "http://localhost:8000/api/analysis?debug=true" `
  -H "Content-Type: application/json" `
  -d "{\"input\":\"unweighted graph shortest path BFS\"}"
```

## Recommendations

```http
POST /api/recommendations
POST /api/v1/recommendations
```

Request body：

```json
{
  "problemText": "Find the shortest path in an unweighted graph.",
  "mode": "hybrid",
  "topK": 3
}
```

`mode` 可為：

```text
hybrid
vector
graph
```

Recommendations endpoint 保留原 demo contract，避免破壞既有 frontend 與測試。

## Ingestion CLI

```powershell
python -m backend.app.ingestion build --input data/raw --processed data/processed --target all
```

`--target` 可為：

```text
json
bm25
qdrant
neo4j
all
```

本機 fallback：

```powershell
python -m backend.app.ingestion build --input data/raw --processed data/processed --target all --allow-fallback
```

當 target 需要 Qdrant 或 Neo4j 但服務不可用，且沒有傳入 `--allow-fallback`，CLI 會以非 0 exit code 失敗並提示啟動 Docker 或改用 fallback。

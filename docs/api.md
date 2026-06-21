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

也支援：

```text
problemText
statement
code
problemId
```

`problemId` 若明確指定但找不到，回 `404`。沒有 `problemId` 的一般文字輸入會走 query search，不會回 404。

空輸入回 `400`。

### Analysis Response

既有欄位保留：

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

新增欄位：

```text
retrievalTrace
evidenceBundle
contextPreview
```

`contextPreview` 僅在 debug mode 回傳：

```http
POST /api/analysis?debug=true
```

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

此 endpoint 保留既有 demo recommendation contract，供前端與舊測試相容。

## Ingestion CLI

```powershell
python -m backend.app.ingestion build --input data/raw --processed data/processed --target all
```

`--target`：

```text
json
bm25
qdrant
neo4j
all
```

Docker 不可用但需要本機 artifact：

```powershell
python -m backend.app.ingestion build --input data/raw --processed data/processed --target all --allow-fallback
```

未提供 `--allow-fallback` 且 Qdrant / Neo4j 無法連線時，CLI 會回非零 exit code 並提示啟動 Docker 或加上 `--allow-fallback`。

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
  "input": "unweighted graph shortest path BFS",
  "mode": "hybrid",
  "topK": 3
}
```

相容欄位：

```text
input
problemText
statement
code
problemId
mode
topK
```

行為：

- 空輸入回 `400`。
- 若指定 `problemId` 且不存在，回 `404`。
- 若沒有指定 `problemId`，一般文字會走 query search。
- `mode` 可為 `hybrid`、`vector`、`graph`，預設 `hybrid`。
- `topK` 控制最後候選數，預設 `5`。
- `similarProblems` 來自所選 mode 的最後 reranked candidates；命中題目本身會放在 `matchedProblem`，不會重複出現在 `similarProblems`。
- 當所選 mode 沒有相似候選時，`similarProblems` 會是空陣列，不會回退到無關的 demo 相似題。

### Analysis Response

`status` is an analysis outcome, not an HTTP error indicator:

- `ok` means the backend found a programming problem, submitted-code feature,
  exact problem, or retrieval evidence that supports analysis.
- `unsupported` is an intentional abstention. `requiredConcepts`,
  `similarProblems`, `evidencePaths`, and `evidenceBundle.graphPaths` are empty;
  `abstentionReason` explains why the input is outside the supported scope.

An oversized input is rejected before analysis with HTTP `413` and a `detail`
object whose `code` is `input_too_large`.

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
retrievalBackend
```

`contextPreview` 只在 debug mode 回傳：

```http
POST /api/analysis?debug=true
```

When present, it may include enriched evidence such as answer,
solutionHints, difficulty, constraints, and graphPaths.

### Retrieval Trace

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
      "candidateSource": "qdrant",
      "payload": {
        "title": "Rotting Oranges",
        "documentSource": "LeetCode",
        "sourceId": "994",
        "answer": "Use BFS from all initially rotten oranges.",
        "solutionHints": ["Push all sources first."],
        "difficulty": "Medium",
        "constraints": ["1 <= m, n <= 10"]
      }
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
      "candidateSource": "bm25_index",
      "payload": {
        "title": "Rotting Oranges",
        "documentSource": "LeetCode",
        "sourceId": "994",
        "answer": "Use BFS from all initially rotten oranges.",
        "solutionHints": ["Push all sources first."],
        "difficulty": "Medium",
        "constraints": ["1 <= m, n <= 10"]
      }
    }
  ],
  "fusionScores": [],
  "rerankerScores": []
}
```

`candidateSource` is only added when `debug=true`. Non-debug responses keep the
existing `retrievalTrace` shape and omit `retrievalBackend`.

When available, debug traces also include `compatibilityWarnings` for bounded
store-adapter diagnostics:

```json
{
  "compatibilityWarnings": [
    {
      "adapter": "qdrant",
      "severity": "warning",
      "message": "Qdrant client 1.18.0 is outside the supported server 1.15.3 minor range."
    }
  ]
}
```

This field appears only with `debug=true`. It is diagnostic and non-fatal: a
compatibility warning does not fail the request or change retrieval behavior.

Debug candidates and graph paths include `scoreMeta`. Its `stage` and
`displayLabel` identify the scoring stage, and `comparableAcrossStages` states
whether that score can be compared with another stage. Candidate provenance can
also include `chunkEvidence` with `available`, `complete`, `missingSources`,
and `unavailableReason`.

Store-backed vector and BM25 normalized candidate `payload` values can include
enriched fields such as answer, solutionHints, difficulty, constraints,
examples, editorial, documentSource, sourceId, title, problemType, and concepts.
Raw `storePayload` may retain the processed/store field name `source`.
`stores` mode uses processed runtime documents from `PROCESSED_PROBLEMS_PATH`
for the online candidate set.

Store-backed graph paths use the same display summary as local graph paths:
`input -> linked entity -> problem`. When the graph store returns a raw path, the
raw `nodes` and `relations` are preserved under `storePath`. Graph paths may
also include a `rationale` used by debug `contextPreview`.

Each graph path identifies how it was produced with `graphPathOperation`:
`candidate_retrieval` for a retrieved candidate path or `exact_expansion` for
evidence expanded from an exact problem match. Nodes use the layers `problem`,
`chunk`, `concept`, `code_feature`, `pattern`, and `source`. Each relation has a
typed `type` and local-confidence `weight`. `pathScoring` uses
`strategy=weighted_layered_path_v1` and supplies the path `score` plus its
`components`; the top-level graph path `score` is the same value as
`pathScoring.score`.

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

`evidenceBundle.similarProblems` follows the same selected-mode candidate set
as top-level `similarProblems`. `ContextBuilder` includes the `相似題` section
only when this array is non-empty.

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

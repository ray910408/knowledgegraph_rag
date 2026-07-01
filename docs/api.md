# API 契約

## 健康檢查

```http
GET /api/health
GET /api/v1/health
```

回應：

```json
{ "status": "ok" }
```

## 分析 API

```http
POST /api/analysis
POST /api/v1/analysis
```

請求內容：

```json
{
  "input": "給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。",
  "mode": "hybrid",
  "topK": 3
}
```

相容輸入欄位：

```text
input
problemText
statement
code
problemId
mode
topK
```

行為規則：

- 空輸入回 `400`。
- 若指定 `problemId` 且找不到，回 `404`。
- 若沒有指定 `problemId`，一般文字會走 query search。
- `mode` 可為 `hybrid`、`vector`、`graph`，預設 `hybrid`。
- `topK` 控制最後候選數，預設 `5`。
- `matchedProblem` 專門放 exact problem hit，不會重複塞進 `similarProblems`。
- 當所選 mode 沒有相似候選時，`similarProblems` 會是空陣列，不會回退到無關的示意題目。

## 分析回應

`status` 代表分析結果，不是 HTTP 錯誤碼：

- `ok`：後端找到了程式題、程式碼特徵、exact problem，或足夠的 retrieval evidence。
- `unsupported`：刻意 abstain。此時 `requiredConcepts`、`similarProblems`、`evidencePaths`、`evidenceBundle.graphPaths` 會是空的，原因在 `abstentionReason`。

過大的輸入會在分析前被拒絕，回 `413`，且 `detail.code` 會是 `input_too_large`。

`zh-Hant` 的 `abstentionReason` 必須維持下列精確字串：

- `未偵測到程式題、程式碼、演算法概念或可靠檢索證據。`
- `輸入超出目前支援範圍，請提供程式題敘、題號、程式碼或已知演算法概念。`

頂層欄位：

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

### `/api/analysis` 回應 scope 規則

- 頂層 `similarProblems` 由 `EvidenceBuilder` scope selection 後的 `evidenceBundle.similarProblems` 派生，不得在 scope selection 後直接讀取 raw `reranked_candidates`。
- exact problem match 時，`matchedProblem` 會固定為 evidence anchor；`algorithmEvidence`、`dataStructureEvidence`、`patternEvidence`、`techniqueEvidence`、`commonMistakes` 與 `solvingHints` 必須以此 anchor 與 canonical concept scope 為準。顯示用相似題是 optional，可保留同 scope 的 filtered display candidates，但不得混入 raw unrelated candidates。
- concept-only query（例如 `DP`、`dynamic programming`、`動態規劃`）必須把 `DP` 與 `Dynamic Programming` 視為同一個 canonical concept。
- concept-only query 的 `requiredConcepts` 只能來自 query seeds 與 filtered evidence，不得從 raw reranked candidates 繼承無關概念。

只有在 `debug=true` 才會多出：

```text
contextPreview
retrievalBackend
```

## `retrievalTrace.queryUnderstanding`

`queryUnderstanding` 現在是多語查詢理解的主要輸出，欄位如下：

```text
originalQuery
normalizedQuery
inputKind
intent
keywords
queryLanguage
exactTerms
lowWeightTerms
conceptSeeds
expandedTerms
queryVariants
codeFeatures
```

說明：

- `queryLanguage`：`zh-Hant`、`en`、`mixed`。
- `keywords`：供 trace 與檢索使用的實際關鍵詞，中文查詢會保留中文詞組。
- `exactTerms`：從查詢中直接抓到的詞組或 alias。
- `lowWeightTerms`：像 `給定`、`請`、`找出` 這類低權重提示詞。
- `conceptSeeds`：依規則推導出的概念，例如 `BFS`、`Queue`、`Shortest Path`。
- `expandedTerms`：加入英文 alias 後的擴展詞。
- `queryVariants.bm25`：原始查詢加英文 alias。
- `queryVariants.vector`：優先使用英文語意化改寫，否則退回擴展詞。
- `queryVariants.graphSeeds`：圖檢索直接可用的 concept 或 pattern entity IDs。

範例：

```json
{
  "queryUnderstanding": {
    "originalQuery": "給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。",
    "normalizedQuery": "給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。",
    "inputKind": "problem",
    "intent": "problem_search",
    "keywords": ["無權圖", "最短步數", "起點", "終點"],
    "queryLanguage": "zh-Hant",
    "exactTerms": ["無權圖", "最短步數", "起點", "終點"],
    "lowWeightTerms": ["給定", "請", "找出", "從", "到"],
    "conceptSeeds": ["Shortest Path", "BFS", "Graph Traversal", "Queue", "Visited Array"],
    "expandedTerms": [
      "unweighted graph",
      "shortest path",
      "shortest steps",
      "BFS",
      "breadth first search",
      "breadth-first search",
      "Queue",
      "Visited Array",
      "visited array",
      "visited set",
      "Graph Traversal",
      "graph traversal",
      "source",
      "target",
      "start",
      "end"
    ],
    "queryVariants": {
      "bm25": "給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。 unweighted graph shortest path shortest steps BFS breadth first search breadth-first search Queue Visited Array visited array visited set Graph Traversal graph traversal source target start end",
      "vector": "find the shortest path in an unweighted graph from source to target using bfs and a queue",
      "graphSeeds": [
        "concept:shortest-path",
        "concept:bfs",
        "pattern:graph-traversal",
        "concept:queue",
        "concept:visited-array"
      ]
    }
  }
}
```

## `entityLinking`

`entityLinking` 會整合三種來源：

- `matchedProblem`
- 程式碼特徵對應的 `code_feature:*`
- `queryVariants.graphSeeds` 直接帶入的概念種子

其中 concept seed 連結會標上：

```text
matchedBy = concept_seed
confidence = 1.0
```

程式碼特徵則會額外帶：

```text
matchedBy = code_feature
codeFeatureNodeId
```

## 候選與 debug trace

`retrievalTrace` 會包含：

```text
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

注意事項：

- `candidateSource` 只在 `debug=true` 時出現。
- `vectorCandidates`、`bm25Candidates` 的 `payload` 可能包含 `answer`、`solutionHints`、`difficulty`、`constraints`、`examples`、`editorial`、`documentSource`、`sourceId`、`title`、`problemType`、`concepts`、`displayText`、`searchText`。
- raw store payload 會保留在 `storePayload`。
- `bm25Candidates` 只會保留 `score > 0` 的候選。
- 所有候選與 graph path 的分數都會帶 `scoreMeta`，不同 stage 的分數不能直接互相比較。
- `searchText` 是 index-only lane，debug payload 可能會保留它做驗證，但 UI 與人工閱讀內容應優先使用 `text` / `displayText`。

若有 store adapter 相容性診斷，debug trace 也會包含：

```json
{
  "compatibilityWarnings": [
    {
      "adapter": "qdrant",
      "severity": "warning",
      "message": "..."
    }
  ]
}
```

### `retrievalTrace.rawGraphPaths`

`retrievalTrace.rawGraphPaths` 只在 `debug=true` 時輸出，是 graph search 尚未被 reranking 與 evidence selection 裁切前的原始路徑集合。一般回應 contract 不應依賴此欄位；UI 與 prompt context 應使用 `evidenceBundle.graphPaths`。

## `evidenceBundle`

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

用途：

- `similarProblems`：只來自 `EvidenceBuilder` 選出的 display candidates；exact match 時，候選的 canonical concept/problem-type scope 必須完整包含於 matched scope，且不得重複 exact match。
- `graphPaths`：保留 `nodes`、`relations`、`pathSource`、`graphPathOperation`、`pathScoring`、`scoreMeta`。
- `techniqueEvidence`：放 `Visited Array` 這種不是演算法名稱但很重要的技巧證據。
- `matchedProblem`：exact problem 命中時，可在 evidence layer 再顯示一次同一筆 canonical record。

### `evidenceBundle.graphPaths` 與 selected evidence

`evidenceBundle.graphPaths` 是 post-rerank pruned graph paths，只保留回應採用的 graph evidence；完整原始路徑僅能從 debug-only `retrievalTrace.rawGraphPaths` 觀察。

matched evidence 會使用 `problemCard`、statement、solution 與 hints 組成人類可讀摘要；similar evidence 會使用 `problemCard` 加上 `matchedChunk` 說明命中的 chunk。

## Debug mode

啟用方式：

```http
POST /api/analysis?debug=true
```

此時回應可額外帶：

- `contextPreview`
- `retrievalBackend`
- `candidateSource`
- `compatibilityWarnings`

`contextPreview` 會使用 enriched payload、display/context lane 與 graph path rationale 組成 LLM prompt context，不會直接把 `searchText` alias expansion 回傳給前端。非 debug response 不會回傳它。

`ContextBuilder` 必須從 `displayText` / display lane 與 evidence lane 建立 prompt context，不得讀取 chunk `searchText`。`contextPreview` 因此只能顯示乾淨文字、selected evidence 與已 pruned graph paths。

## Recommendations

```http
POST /api/recommendations
POST /api/v1/recommendations
```

請求內容：

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

Recommendations API 保留既有 demo 契約，避免破壞現有 frontend 與測試。

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

若 target 需要 Qdrant 或 Neo4j，但服務不可用且沒有帶 `--allow-fallback`，CLI 會以非 0 exit code 明確失敗。

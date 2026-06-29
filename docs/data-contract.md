# 資料契約

## 原始題目資料

### 識別欄位

- `id`：跨系統的 canonical identifier，例如 `leetcode-1091`、`uva-10653`。
- `sourceId`：來源站點自己的 identifier，例如 `1091`、`10653`。

所有 join 都以 `id` 為主；`sourceId` 保留給顯示與來源站點參照。

`data/raw/*.json` 可以是單一 object、array，或 `{ "problems": [...] }`。

Repo 內建 `data/raw/programming_problems.json` 作為可執行 seed fixture。檔案必須保持 UTF-8 可讀的繁體中文，不可把中文內容改寫成 `\uXXXX` escape。

### 必填欄位

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

### 可選欄位

```text
difficulty
constraints
examples
editorial
```

### 範例

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

清理與標準化後的題目資料。去重 key 為 `source + sourceId`。

### `chunks.json`

欄位：

```text
id
problemId
kind
text
displayText
searchText
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
problem_card
statement
constraints
examples
hints
solution
common_mistakes
```

`text` 在 rollout 1 中維持為 `displayText` 的 legacy alias。

補充契約：

- `displayText` 是乾淨的 display/context lane，提供 UI、evidence、`contextPreview` 與其他人類可讀輸出使用。
- `searchText` 是 index-only lane，只提供 BM25 與向量 embedding 使用。
- `text == displayText`，避免既有依賴在 rollout 期間破壞。
- template-derived `commonMistakes` 不得進入 `searchText`。若 `commonMistakesSource == "template"`，該段不得被納入索引文字。

### Display / search lane contract

`displayText` 是 display/context lane，供 UI、evidence、`contextPreview` 與 prompt context 使用。`searchText` 是 index-only lane，只供 BM25 與 vector embedding 使用；`ContextBuilder` 不得讀取 chunk `searchText`，也不得把 alias expansion 或 template-derived `commonMistakes` 外洩到顯示內容。

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
technique
pattern
concept
```

這個分支的 `aliases` 會加入雙語概念別名，例如：

- `BFS`：`廣度優先搜尋`、`廣搜`、`breadth first search`
- `Queue`：`佇列`、`隊列`
- `Visited Array`：`拜訪陣列`、`visited 陣列`、`visited set`
- `Graph Traversal`：`圖論遍歷`、`圖遍歷`

`Visited Array`、`拜訪陣列` 與 `visited 陣列` 都屬於 `technique` taxonomy，不應正規化成 `data_structure`。

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

`REQUIRES` 與 `HAS_PATTERN` 是 ingestion、store adapter 與 runtime graph path 都必須接受並保留的 relation type。若遇到未知 relation，normalized relation 必須保留 `normalizedFrom`，讓 debug trace 可回溯原始 type。

### `bm25_index.json`

每個 document 會包含：

```text
id
text
problemId
tokens
payload
```

欄位意義：

- `text`：離線組出的 bilingual `searchText`，包含 `problemId`、`source`、`sourceId`、`title`、`problemType`、`concepts`、concept alias、原始 chunk 顯示文字。
- `tokens`：由 `shared_multilingual_tokens()` 產生，會先保留中文精確詞組，再補 ASCII tokens。
- `payload`：原始 chunk mapping，保留乾淨顯示內容與 enriched evidence 欄位。

補充契約：

- `bm25_index.json.documents[*].text == chunk.searchText`。
- `documents[*].payload.displayText` 與 `documents[*].payload.text` 必須保持乾淨顯示文字，不回寫 alias expansion。

`payload` 會帶：

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

對外正規化後，候選 `payload.source` 會映射成 `documentSource`，原始 store 內容則保留在 `storePayload`。

BM25 candidate stores 只回傳 `score > 0` 的候選；`score <= 0` 的文件不得進入 fusion。

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

補充契約：

- Qdrant vectors 一律由 `searchText` 產生，不直接使用 `displayText` / `text`。
- `records[*].payload.searchText` 會保留 index lane，方便驗證與除錯。

注意：

- 向量不是對 `chunk.text` 直接做 embedding。
- 向量是對跟 BM25 相同的 bilingual `searchText` 做 embedding。
- `payload` 仍保留 chunk 原始欄位，供前端顯示與證據整理使用。

### `neo4j_graph.json`

欄位：

```text
entities
relations
```

這份 artifact 可作為 Neo4j import 與 debug 的中介格式。

## `retrievalTrace` 契約

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

### `queryUnderstanding`

`queryUnderstanding` 內含：

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

其中 `queryVariants` 再包含：

```text
bm25
vector
graphSeeds
```

契約重點：

- `keywords` 對中文查詢會保留中文詞組，不只剩 ASCII token。
- `queryLanguage` 會明確標示 `zh-Hant`、`en`、`mixed`。
- `graphSeeds` 是 entity IDs，例如 `concept:bfs`、`concept:queue`、`pattern:graph-traversal`。
- 所有擴展都是 additive，不會用翻譯結果覆蓋原始查詢。

### `entityLinking`

`entityLinking` 每筆可能帶：

```text
entityId
name
type
confidence
matchKind
matchedBy
codeFeatureNodeId
```

`matchedBy` 可能值：

```text
concept_seed
code_feature
```

### 候選 payload 與 provenance

store 模式候選的 `payload` 可能包含：

```text
documentSource
sourceId
answer
solutionHints
difficulty
constraints
examples
editorial
title
problemType
concepts
storeCandidateId
storePayload
chunkEvidence
```

`chunkEvidence` 內含：

```text
available
complete
missingSources
unavailableReason
```

`bm25Candidates` 只保留 `score > 0` 的候選，避免零分列也進入 fusion。

### 分數契約

候選與 graph path 的分數都會附帶 `scoreMeta`。呼叫端必須先看 `scoreMeta.stage` 與 `scoreMeta.comparableAcrossStages`，不能把 BM25、vector、graph path、fusion、reranker 分數直接混比。

## Graph Path 契約

store 模式 graph path 會同時保留穩定形狀與原始路徑：

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

規則：

- `nodes` / `relations`：穩定的 public shape。
- `storePath`：保留 Neo4j 原始路徑。
- `graphPathOperation`：`candidate_retrieval` 或 `exact_expansion`。
- node layer 可為 `problem`、`chunk`、`concept`、`code_feature`、`pattern`、`source`。

## 分析回應契約

頂層分析回應固定包含：

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

只有 `debug=true` 時才會多出：

```text
contextPreview
retrievalBackend
```

`matchedProblem` 代表 exact canonical problem hit，會跟 `similarProblems` 分開。

## `evidenceBundle` 契約

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

重點：

- `similarProblems` 只來自最終 selected-mode candidates。
- `graphPaths` 可帶 `nodes`、`relations`、`score`、`rationale`、`storePath`、`pathSource`、`graphPathOperation`、`pathScoring`、`scoreMeta`。
- `evidenceBundle.graphPaths` 是 post-rerank pruned graph paths，只保留 selected evidence 使用的路徑；未裁切的原始路徑只會出現在 debug-only `retrievalTrace.rawGraphPaths`。
- matched evidence 必須使用 `problemCard`、statement、solution 與 hints；similar evidence 必須使用 `problemCard` 與 `matchedChunk`，讓使用者看見被比對的題目摘要與實際命中片段。
- `techniqueEvidence` 會放 `Visited Array` 這類技巧證據。

## `contextPreview` 契約

`contextPreview` 是送入 `LLMProvider` 前整理好的 prompt context，只在 `debug=true` 時回傳。

它可能使用：

- enriched candidate payload 的 `answer`、`solutionHints`、`difficulty`、`constraints`
- graph path 的 `rationale`

Context guardrail：

- `ContextBuilder` 必須只讀取 display/context lane 與 evidence lane。
- `ContextBuilder` 不得直接消費任何 chunk `searchText`。
- 當 `similarProblems` 為空時，`ContextBuilder` 會直接省略 `相似題` 區段，不會輸出空標題。

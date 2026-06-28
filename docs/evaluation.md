# 驗證與評估

評估目標是確認 hybrid retrieval 比單一路徑更穩定，也更能解釋為什麼推薦這些演算法、資料結構與圖譜路徑。

## 基準組

- 向量檢索：只靠語意相似度排序候選。
- 圖譜檢索：只靠既有 problem / concept / pattern 關係找證據。
- 混合檢索：綜合向量分數、graph path 支持與概念重疊。
- 無 LLM：只輸出 evidence bundle，不做自然語言回答。

## 指標

- Top-1、Top-3、Top-5 演算法命中率。
- Top-1、Top-3、Top-5 資料結構命中率。
- pattern 命中率。
- 對「容易誤判但應排除」案例的排除正確率。
- 30 題人工檢查樣本的 evidence path 可解釋性。

## 控制條件

- 向量檢索與混合檢索使用同一組 embedding model 與候選池。
- `mode=vector`、`mode=graph`、`mode=hybrid` 都要只看該 mode 的最終候選，不可偷吃別的路徑。
- `stores` 模式 retrieval 測試要跟 local fallback fixtures 維持可比性，避免因為 store adapter 不同而改變預期答案。
- runtime stores 檢查需使用 `PROCESSED_PROBLEMS_PATH=data/processed/problems.json`，確保線上 API 路徑真的讀到 processed runtime documents。
- 這一階段的 Query Understanding 仍維持 rule-based。若未來要加翻譯或 LLM，必須是 additive，不可拿翻譯結果覆蓋原始 query。
- 評估時記錄錯誤類型，例如 graph edge 缺失、label 錯誤、embedding 弱匹配、問題敘述含糊、LLM 用詞問題。

## 這個分支的最低驗收

- 中文自然語言 query 能產生非空 `keywords`、`conceptSeeds`、`expandedTerms`、`queryVariants`。
- `BM25SearchService` 使用 `queryVariants["bm25"]`，且不返回 `score <= 0` 的候選。
- `VectorSearchService` 使用 `queryVariants["vector"]`。
- `EntityLinkingService` 能直接吃 `graphSeeds`，`GraphSearchService` 沒有 exact matched problem 也能走概念種子。
- ingestion 產出的 `bm25_index.json` 與 `qdrant_vectors.json` 都用 bilingual `searchText`，不是只靠原始 chunk 文字。

## 驗證指令

```powershell
python -m ruff check .
python -m pytest tests/backend
cd frontend
npm.cmd run build
```

如果要做 stores-mode 靜態檢查：

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\quick-start.ps1 -Check -Stores
```

## 迴歸案例

| 案例 | 期望契約 |
|---|---|
| 後端停止 | 前端必須顯示明確錯誤，不能偽造成功的 API 回應。 |
| Exact UVA query | `UVA-10653 - Bombs! NO they are Mines!!`、裸 `10653`、以及部分 UVA 標題都要解析成 `matchedProblem.id=uva-10653`；chunk evidence 要涵蓋 statement、answer、hints；graph evidence 要有 canonical path、operation label、layer、type、weight、path scoring。 |
| 等價 BFS 程式碼 | 等價的 C++ 與 Python BFS 程式碼都要連到 `BFS`、`Queue`、`Visited Array`，並透過 `code_feature:*` 節點留下證據。 |
| 中文最短路徑 query | `給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。` 要產生 `queryLanguage=zh-Hant`，`keywords` 含 `無權圖`、`最短步數`，`conceptSeeds` 含 `BFS`、`Queue`、`Shortest Path`、`Graph Traversal`。 |
| BM25 多語 query | 中文 query 的 `queryVariants.bm25` 必須同時保留原始中文與英文 alias，並回傳非空 BM25 candidates。 |
| Vector semantic variant | 中文 query 的 `queryVariants.vector` 必須優先使用語意化英文改寫，至少能看出 `unweighted graph`、`source`、`target`、`bfs`、`queue` 的語意。 |
| Graph seeds | 中文 query 的 `queryVariants.graphSeeds` 必須包含 `concept:bfs`、`concept:queue`、`concept:shortest-path` 等 entity IDs，Graph Search 要能回傳 graph paths。 |
| 無關文字 | 晚餐、天氣、捷運等無關輸入要回 `status=unsupported`，不能亂給 BFS 概念或推薦題目。 |
| 空白輸入 | 純空白輸入必須被拒絕。 |
| 過大輸入 | 超大輸入要回 HTTP `413` 與 `input_too_large`。 |
| 分數標籤 | 所有對外顯示的 candidate 與 graph path 分數都要帶 `scoreMeta`，不能跨 stage 直接比較。 |
| 識別欄位 join | 回應記錄要使用 canonical `id`，並保留 source-local `sourceId`。 |

## 手動 smoke query

```text
給定一張無權圖與起點、終點，請找出從起點到終點的最短步數。需要並明確使用哪些演算法與資料結構。
```

手動預期：

- `keywords` 非空
- `conceptSeeds` 含 `BFS`、`Queue`、`Shortest Path`、`Graph Traversal`
- `bm25Candidates` 非空
- `graphPaths` 非空
- 回答明確說明無權圖最短步數應使用 BFS

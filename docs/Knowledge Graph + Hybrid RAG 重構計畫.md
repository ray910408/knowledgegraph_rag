# Knowledge Graph + Hybrid RAG 重構計畫

## Store-backed retrieval update

- Online retrieval services support injected `VectorStore`, `BM25Store`, and
  `GraphStore` implementations while preserving the local document fallback.
- Store-backed graph trace summaries remain `input -> linked entity -> problem`;
  raw store nodes and relations are preserved under `storePath`.
- FastAPI runtime store mode now wires Qdrant, Neo4j, BM25Store, and processed
  runtime documents into the live API when `RETRIEVAL_BACKEND=stores`.
- `PROCESSED_PROBLEMS_PATH=data/processed/problems.json` supplies the runtime
  document set in stores mode; local mode still uses the local fallback
  documents.
- Qdrant and BM25 candidate payloads now carry enriched evidence fields such as
  answer, solutionHints, difficulty, constraints, examples, editorial, source,
  sourceId, title, problemType, and concepts.
- `/api/analysis` now honors `mode` and `topK`, keeps exact problem matches in
  `matchedProblem`, and derives `similarProblems` from the selected-mode final
  candidates only.
- Empty similar-problem evidence no longer renders stale UI or context-preview
  headings.
- Query Understanding remains rule-based in this round, so retrieval refactoring
  is not mixed with LLM behavior changes.

## Summary

本次重構把原本可測的 demo scaffold 拆成正式的 Knowledge Graph + Hybrid RAG 架構，並明確分成兩條流程：

- 離線建庫流程：處理 CPE / LeetCode raw data、資料清理與標準化、chunking、entity/relation extraction、processed JSON、BM25 index、Qdrant vector records、Neo4j graph records。
- 線上查詢流程：處理 Query Understanding、Query Embedding、Entity Linking、BM25 Search、Qdrant Vector Search、Neo4j Graph Search、Hybrid Fusion、Reranker、Evidence Builder、Context Builder、LLM Response Generator。

實作前已確認 Git 狀態，並在空 `.git` 目錄下重新初始化 repository、建立 initial commit，再從 feature branch 分階段實作。

## Key Changes

- 新增 `backend/app/ingestion/` 與 CLI：
  `python -m backend.app.ingestion build --input data/raw --processed data/processed --target all`
- `--target` 支援 `json`、`bm25`、`qdrant`、`neo4j`、`all`。
- ingestion pipeline 會輸出 problems、chunks、entities、relations、BM25 index、Qdrant vector records、Neo4j graph records 與 manifest。
- 將線上 retrieval 拆成 Query Understanding、Embedding、Entity Linking、BM25、Vector、Graph、Fusion、Reranker、Evidence、Context、LLM response 等可單測服務。
- 保留 `/api/analysis`、`/api/v1/analysis`、`/api/recommendations`、`/api/v1/recommendations`。
- Analysis response 保留既有欄位，新增 `retrievalTrace`、`evidenceBundle`、debug-only `contextPreview`。
- 新增 provider / adapter 介面：`EmbeddingProvider`、`LLMProvider`、`GraphStore`、`VectorStore`、`BM25Store`。
- 測試預設使用 `DeterministicMockEmbeddingProvider` 與 mock/in-memory stores。正式模型名稱保留 `BAAI/bge-m3`。
- Qdrant 與 Neo4j adapter 已拆出，測試不依賴 Docker。
- Frontend 改成「輸入 -> 查詢理解 -> 三路檢索 -> fusion/rerank -> evidence/context -> 回答」的 trace view，並保留 mock fallback。
- 文件更新：`README.md`、`docs/architecture.md`、`docs/data-contract.md`、`docs/api.md`。

## Public Interfaces

### Ingestion CLI

```powershell
python -m backend.app.ingestion build --input data/raw --processed data/processed --target all
```

本機 demo：

```powershell
python -m backend.app.ingestion build --input data/raw --processed data/processed --target all --allow-fallback
```

### Raw Problem Schema

保留欄位：

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

新增可選欄位：

```text
difficulty
constraints
examples
editorial
```

### Analysis Response

新增可選欄位：

```text
retrievalTrace
evidenceBundle
contextPreview
```

`contextPreview` 只在 `POST /api/analysis?debug=true` 或 `POST /api/v1/analysis?debug=true` 回傳。

`problemId` 行為定義為：指定題號不存在時回 `404`；未指定題號時，一般文字走 query search。

## Test Plan

- 保留並更新既有 backend 測試，確保舊 endpoint contract 不破。
- 新增 ingestion tests：資料清理、chunking、mock extraction、processed JSON、去重。
- 新增 adapter tests：Qdrant / Neo4j 不可用時不依賴 Docker，mock / in-memory adapter 可正常通過。
- 新增 retrieval tests：BM25、Vector、Graph、Fusion、Reranker、Evidence Builder、Context Builder。
- 新增 API tests：`retrievalTrace`、`evidenceBundle`、`debug=true` 的 `contextPreview`、空輸入 400、未知 `problemId` 404。
- 新增 frontend smoke：`npm.cmd run build` 通過，主要畫面不再出現中文亂碼。

## Assumptions

- 採用完整重構。
- 正式 demo 預設使用 Docker 的 Neo4j + Qdrant。
- 測試仍使用 mock / in-memory，不要求 Docker。
- LLM 與 extraction 先做 provider interface + mock，不在本輪要求真實 API key。
- 本輪不做爬蟲，不自動抓 CPE / LeetCode，只處理使用者放進 `data/raw/` 的資料。
- 本輪不做完整 accepted code generation。
- 程式碼錯誤診斷可保留為加分功能，不阻塞主流程。

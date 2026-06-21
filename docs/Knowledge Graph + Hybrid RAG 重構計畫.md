# Knowledge Graph + Hybrid RAG 重構計畫

## Summary

將目前 demo scaffold 重構為兩條正式流程：

- 離線建庫：清理 CPE/LeetCode raw data、標準化、chunking、entity/relation extraction、processed JSON、BM25 index、Qdrant vector records、Neo4j graph records。
- 線上查詢：Query Understanding、Query Embedding、Entity Linking、BM25 Search、Qdrant Vector Search、Neo4j Graph Search、Hybrid Fusion、Reranker、Evidence Builder、Context Builder、LLM Response Generator。

實作前先確認 Git 狀態。若 `C:\knowledgegraph_rag\.git` 仍是空目錄或 repo 無效，停止並先建立可追蹤的 initial commit。

## Key Changes

- 新增 `backend/app/ingestion/` 與 CLI：
  `python -m backend.app.ingestion build --input data/raw --processed data/processed --target all`
- `--target` 支援 `json`、`bm25`、`qdrant`、`neo4j`、`all`；Docker DB 不可用時清楚報錯，測試使用 mock/in-memory。
- 將 `HybridRetrievalService` 拆成 Query Understanding、Embedding、Entity Linking、BM25、Vector、Graph、Fusion、Reranker、Evidence、Context、LLM response services。
- 保留既有 `/api/analysis`、`/api/v1/analysis`、`/api/recommendations`、`/api/v1/recommendations`，只新增可選欄位。
- 新增 provider/adapter：`EmbeddingProvider`、`LLMProvider`、`GraphStore`、`VectorStore`、`BM25Store`。
- 預設測試使用 `DeterministicMockEmbeddingProvider` 與 `MockLLMProvider`；正式模型名稱保留 `BAAI/bge-m3`。
- 前端修正中文亂碼，改為展示「輸入 → 查詢理解 → 三路檢索 → fusion/rerank → evidence/context → 回答」流程。
- 更新 `README.md`、`docs/architecture.md`、`docs/data-contract.md`、`docs/api.md`。

## Public Interfaces

- Ingestion CLI：
  `python -m backend.app.ingestion build --input data/raw --processed data/processed --target all`
- Raw problem schema 保留既有欄位，新增可選 `difficulty`、`constraints`、`examples`、`editorial`。
- Analysis response 保留既有欄位，新增：
  `retrievalTrace`、`evidenceBundle`、`contextPreview`。
- `contextPreview` 只在 `POST /api/analysis?debug=true` 或 `POST /api/v1/analysis?debug=true` 回傳。
- 明確行為：指定 `problemId` 不存在時回 `404`；未指定題號的一般文字輸入走 query search。

## Test Plan

- 保留並更新現有 23 個 backend 測試，確保舊 API contract 不破。
- 新增 ingestion tests：清理、chunking、mock extraction、processed JSON、去重。
- 新增 adapter tests：Qdrant/Neo4j 不可用不影響單元測試，mock/in-memory 可通過。
- 新增 retrieval tests：BM25、Vector、Graph、Fusion、Reranker、Evidence Builder、Context Builder。
- 新增 API tests：`retrievalTrace`、`evidenceBundle`、`debug=true` 才回 `contextPreview`、空輸入 400、未知 `problemId` 404。
- 新增 frontend smoke：`npm.cmd run build` 通過，主要畫面無中文亂碼。

## Assumptions

- 採用完整重構。
- 正式 demo 預設使用 Docker Neo4j + Qdrant。
- 測試不要求 Docker 或真實模型。
- LLM 與 entity/relation extraction 先做 interface + mock。
- 不做爬蟲，不自動抓 CPE/LeetCode。
- 不做完整 accepted code generation。
- 程式碼錯誤診斷保留為加分功能，不阻塞主流程。

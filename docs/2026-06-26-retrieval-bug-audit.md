# 2026-06-26 Retrieval Bug Audit

## 背景

這份文件整理 retrieval / evidence / frontend contract 這一族問題的驗證範圍，起點是使用者列出的 9 個 retrieval 缺陷，並明確擴大到同類型 bug：

1. 中文自然語言 query 的 Query Understanding 關鍵字抽取不足。
2. 中文 query 的 BM25 候選缺失，繁中同義詞與中英 alias 不足。
3. 概念型 graph query 雖能命中概念，但公開 graph path 不夠完整。
4. Hybrid 在 graph / BM25 缺席時缺少明確 degraded-mode 訊號。
5. Chunking 與 evidence payload 需要 problem-aware section 化。
6. Context Builder 對 matched / similar problem 的 evidence 使用不完整。
7. 精準題目的回答不夠題目特化，特別是 `uva-10653`。
8. `commonMistakes` 過度模板化，沒有題目細節。
9. Graph path surface 缺少相似題推理鏈的公開表達。

## 這次出貨對齊的文件事實

目前 repo 中，以下 retrieval contract 已經由 code 與 tests 支撐，文件也已補齊：

- `matchedProblem` 是 exact problem 命中後獨立 surface 的 canonical record，不會混進 `similarProblems`。
- `retrievalTrace.matchedProblem` 可攜帶 exact problem 的 `id`、`sourceId`、`matchKind`、`confidence` 與 seed evidence。
- `evidenceBundle` 除了 `algorithmEvidence`、`dataStructureEvidence`、`patternEvidence` 外，還包含 `techniqueEvidence` 與 `matchedProblem`。
- Graph path contract 包含 `pathSource`、`graphPathOperation`、`pathScoring`，`storePath` 則保留原始圖資料來源。
- frontend contract 會顯示 zh-Hant retrieval copy，並維持 matched problem / evidence / common mistakes 的現有 surface。

## 本輪文件與回歸覆蓋

本輪實際對齊內容：

- `README.md`
  - 新增這份 audit 文件入口，避免文件不可發現。
- `docs/api.md`
  - 補上 top-level `matchedProblem`。
  - 補上 `retrievalTrace.matchedProblem` 的說明。
  - 補上 `techniqueEvidence` 與 `evidenceBundle.matchedProblem`。
- `docs/data-contract.md`
  - 補上 `retrievalTrace.matchedProblem`。
  - 補上 `techniqueEvidence`、`matchedProblem`。
  - 補上 graph path 的 `pathSource`、`graphPathOperation`、`pathScoring`。
- `tests/backend/test_frontend_contracts.py`
  - 補強 zh-Hant copy、UTF-8、hybrid retrieval panel、matched problem / common mistakes 的回歸檢查。

## 驗證

已重新執行：

```powershell
rtk python -m pytest tests/backend/test_frontend_contracts.py tests/backend/test_analysis.py tests/backend/test_online_retrieval_pipeline.py tests/backend/test_contracts_and_providers.py tests/backend/test_runtime_retrieval.py tests/backend/test_ingestion_pipeline.py tests/backend/test_processed_problem_loader.py -q
```

結果：

- `181 passed, 2 warnings in 2.41s`

已重新執行：

```powershell
rtk python -m ruff check tests/backend/test_frontend_contracts.py backend/app/main.py backend/app/retrieval/pipeline.py backend/app/contracts.py
```

結果：

- `All checks passed!`

Frontend build 在這台工作環境目前無法完成，因為 `frontend/node_modules` 不存在。實際探針：

```powershell
rtk npm.cmd run build
```

結果：

- `tsc` 無法找到，代表目前不是程式碼錯誤，而是前端依賴未安裝。

## 結論

這次 ship 的範圍是把 retrieval 相關文件與回歸檢查對齊到目前 repo 的真實 contract，不是再引入新的 runtime 行為。也就是說，這份變更主要修正「文件說的跟現在 code/test 不一致」這種出貨風險，讓後續 PR 與本地 `main` 的狀態一致可驗。

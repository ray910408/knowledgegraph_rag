# 腳本說明

## `quick-start.ps1`

不開 Docker，直接跑本機 fallback demo：

```powershell
.\scripts\quick-start.ps1
```

啟動 `stores` 模式 demo，包含 Neo4j、Qdrant、ingestion、FastAPI、Vite：

```powershell
.\scripts\quick-start.ps1 -Stores
```

只檢查前置條件與路徑，不啟動服務：

```powershell
.\scripts\quick-start.ps1 -Check
.\scripts\quick-start.ps1 -Check -Stores
```

`-Check -Stores` 會額外列出解析後的 Qdrant、Neo4j、`BM25_INDEX_PATH`、`PROCESSED_PROBLEMS_PATH`。其中 `PROCESSED_PROBLEMS_PATH` 預設為 `data/processed/problems.json`。

腳本會啟動：

- FastAPI：`http://127.0.0.1:8000`
- Vite：`http://127.0.0.1:5173`
- Neo4j：`http://localhost:7474` 與 `bolt://localhost:7687`，僅在 `-Stores` 時啟動
- Qdrant：`http://localhost:6333`，僅在 `-Stores` 時啟動

可用 `-BackendPort` 與 `-FrontendPort` 覆蓋埠號。Vite dev proxy 會跟著後端埠號調整。

如果 Docker 服務已經在跑，可加：

```powershell
.\scripts\quick-start.ps1 -Stores -SkipDocker
```

如果 `data/processed/problems.json`、`data/processed/bm25_index.json`、Qdrant、Neo4j 都已經完成 seed，可加：

```powershell
.\scripts\quick-start.ps1 -Stores -SkipIngestion
```

若 Windows execution policy 擋住腳本，可改用：

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\quick-start.ps1 -Check
```

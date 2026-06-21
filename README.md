# Knowledge Graph + Hybrid RAG

程式題庫輔助查詢系統，目標是把 CPE / LeetCode 題庫整理成可檢索的 JSON、BM25 index、Qdrant 向量庫與 Neo4j 知識圖譜，並在線上查詢時合併向量、關鍵字與圖譜證據。

## 架構

系統分成兩條流程：

1. Offline Indexing Pipeline：清理 raw 題庫、標準化 schema、切 chunks、抽取 entities / relations，輸出 BM25、Qdrant records、Neo4j graph records。
2. Online Query Pipeline：做 Query Understanding、Query Embedding、Entity Linking、BM25 / Qdrant / Neo4j 三路檢索、Hybrid Fusion、Reranker、Evidence Builder、Context Builder 與 LLM Response Generator。

詳細圖與元件說明見 [docs/architecture.md](docs/architecture.md)。

## 快速開始

```powershell
python -m pytest tests/backend
cd frontend
npm.cmd run build
```

建立 ingestion artifacts：

```powershell
python -m backend.app.ingestion build --input data/raw --processed data/processed --target all --allow-fallback
```

`--target` 可用：

```text
json
bm25
qdrant
neo4j
all
```

未使用 `--allow-fallback` 時，`qdrant` / `neo4j` / `all` 會嘗試寫入 Docker 服務；服務不可用時 CLI 會清楚回報錯誤。

## API

保留既有 endpoint：

```text
POST /api/analysis
POST /api/v1/analysis
POST /api/recommendations
POST /api/v1/recommendations
```

`POST /api/analysis` 會保留既有 response 欄位，並新增：

```text
retrievalTrace
evidenceBundle
contextPreview
```

`contextPreview` 只會在 `debug=true` 時回傳：

```powershell
curl.exe -X POST "http://localhost:8000/api/analysis?debug=true" `
  -H "Content-Type: application/json" `
  -d "{\"input\":\"unweighted graph shortest path BFS\"}"
```

完整 API contract 見 [docs/api.md](docs/api.md)。

## Docker Services

`docker-compose.yml` 提供：

```text
Neo4j:  http://localhost:7474 / bolt://localhost:7687
Qdrant: http://localhost:6333
```

測試環境不依賴 Docker；backend tests 使用 deterministic mock / in-memory adapters。

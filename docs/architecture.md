# Architecture

本專案已重構為 Knowledge Graph + Hybrid RAG 架構，分成離線建庫與線上查詢兩條流程。

## Offline Indexing Pipeline

```mermaid
flowchart TD
  A["CPE / LeetCode 題庫資料"] --> B["資料清理與標準化"]
  B --> C["Chunking<br/>題目敘述 / 題解 / 概念說明"]
  B --> D["Entity & Relation<br/>Extraction"]
  B --> E["Raw / Processed JSON"]
  C --> F["Embedding Model<br/>bge-m3"]
  F --> G["Vector DB<br/>Qdrant"]
  D --> H["Knowledge Graph<br/>Construction"]
  H --> I["Graph DB<br/>Neo4j"]
```

目前 implementation：

- `backend/app/ingestion/` 提供 CLI 與 artifact builder。
- `RawProblem`、`ProblemChunk`、`EntityRecord`、`RelationRecord` 定義在 `backend/app/contracts.py`。
- `DeterministicMockEmbeddingProvider` 預設產生穩定向量，正式設定仍標示 `BAAI/bge-m3`。
- `QdrantVectorStore`、`Neo4jGraphStore` 已提供 Docker adapter；測試使用 in-memory / fake client。
- `--allow-fallback` 會只寫本機 JSON artifacts，不要求 Docker。

主要輸出：

```text
data/processed/problems.json
data/processed/chunks.json
data/processed/entities.json
data/processed/relations.json
data/processed/bm25_index.json
data/processed/qdrant_vectors.json
data/processed/neo4j_graph.json
data/processed/manifest.json
```

## Online Query Pipeline

```mermaid
flowchart TD
  A["使用者輸入<br/>題目 / 題號 / 關鍵字 / 程式碼"] --> B["Query Understanding<br/>意圖判斷、實體抽取、查詢改寫"]
  B --> C["Query Embedding"]
  B --> D["Entity Linking<br/>對應 Problem / Algorithm / Pattern 節點"]
  B --> E["Keyword Query<br/>BM25 關鍵字"]
  C --> F["Vector Search<br/>Qdrant"]
  D --> G["Graph Search<br/>Neo4j"]
  E --> H["BM25 Search"]
  F --> I["Hybrid Fusion<br/>合併、去重、分數正規化"]
  G --> I
  H --> I
  I --> J["Reranker<br/>重排序候選證據"]
  J --> K["Evidence Builder<br/>整理相似題、圖譜路徑、演算法證據"]
  K --> L["Context Builder<br/>組成 LLM Prompt Context"]
  L --> M["LLM Response Generator"]
  M --> N["輸出<br/>題目理解 / 演算法推薦 / 相似題 / 分層提示 / 常見錯誤"]
```

目前 implementation：

- `backend/app/retrieval/pipeline.py` 拆出可單測服務：
  - `QueryUnderstandingService`
  - `EntityLinkingService`
  - `VectorSearchService`
  - `GraphSearchService`
  - `BM25SearchService`
  - `HybridFusionService`
  - `Reranker`
  - `EvidenceBuilder`
  - `ContextBuilder`
  - `LLMResponseGenerator`
- `POST /api/analysis` 已接上 `retrievalTrace`、`evidenceBundle` 與 debug-only `contextPreview`。
- 舊的 `HybridRetrievalService` 保留，避免破壞既有 recommendations tests。

## Provider / Adapter Boundary

Provider interfaces：

```text
EmbeddingProvider
LLMProvider
```

Store interfaces：

```text
VectorStore
GraphStore
BM25Store
```

Adapter implementations：

```text
InMemoryVectorStore
InMemoryGraphStore
InMemoryBM25Store
QdrantVectorStore
Neo4jGraphStore
```

正式服務可接 Docker Qdrant / Neo4j；測試與本機 demo 可用 mock / in-memory，避免環境耦合。

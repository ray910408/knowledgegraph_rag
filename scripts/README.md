# Scripts

## Quick Start

Run the local fallback demo without Docker:

```powershell
.\scripts\quick-start.ps1
```

Run the store-backed demo with Neo4j, Qdrant, ingestion, FastAPI, and Vite:

```powershell
.\scripts\quick-start.ps1 -Stores
```

Check prerequisites and paths without starting services:

```powershell
.\scripts\quick-start.ps1 -Check
.\scripts\quick-start.ps1 -Check -Stores
```

The script starts:

- FastAPI at `http://127.0.0.1:8000`
- Vite at `http://127.0.0.1:5173`
- Neo4j at `http://localhost:7474` and `bolt://localhost:7687` when `-Stores` is used
- Qdrant at `http://localhost:6333` when `-Stores` is used

Override the app ports with `-BackendPort` and `-FrontendPort`. The Vite
development proxy follows the selected backend port.

Use `-SkipDocker` after Docker services are already running. Use
`-SkipIngestion` after `data/processed/bm25_index.json`, Qdrant, and Neo4j are
already seeded.

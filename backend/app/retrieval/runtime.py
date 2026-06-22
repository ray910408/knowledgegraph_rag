from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from ..adapters.in_memory import InMemoryBM25Store
from ..adapters.neo4j import Neo4jGraphStore
from ..adapters.qdrant import QdrantVectorStore
from ..providers import EmbeddingProvider
from ..stores import BM25Document, SearchCandidate
from .pipeline import OnlineQueryPipeline, RetrievalDocument


RetrievalBackend = Literal["local", "stores"]
JsonMap = dict[str, Any]


class RuntimeRetrievalError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeRetrievalSettings:
    backend: RetrievalBackend = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "programming_chunks"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    bm25_index_path: Path = Path("data/processed/bm25_index.json")


@dataclass(frozen=True)
class RuntimeRetrieval:
    backend: RetrievalBackend
    pipeline: OnlineQueryPipeline
    candidate_sources: dict[str, str]


class JsonBM25Store:
    def __init__(self, documents: Sequence[BM25Document]) -> None:
        self._store = InMemoryBM25Store()
        self._store.index_documents(tuple(documents))

    @classmethod
    def from_path(cls, path: Path) -> JsonBM25Store:
        if not path.exists():
            raise RuntimeRetrievalError(f"BM25 index not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeRetrievalError(f"BM25 index must contain a documents list: {path}")
        raw_documents = payload.get("documents")
        if not isinstance(raw_documents, list):
            raise RuntimeRetrievalError(f"BM25 index must contain a documents list: {path}")

        documents: list[BM25Document] = []
        for raw in raw_documents:
            if not isinstance(raw, dict):
                raise RuntimeRetrievalError(f"BM25 document must be an object in: {path}")
            document_id_value = raw.get("id")
            if document_id_value is None:
                raise RuntimeRetrievalError(f"BM25 document must contain an id in: {path}")
            raw_payload = raw.get("payload")
            if raw_payload is None:
                document_payload = {}
            elif not isinstance(raw_payload, dict):
                raise RuntimeRetrievalError(f"BM25 document payload must be an object in: {path}")
            else:
                document_payload = dict(raw_payload)

            document_id = str(document_id_value)
            text = str(raw.get("text") or "")
            if "problemId" not in document_payload and raw.get("problemId") is not None:
                document_payload["problemId"] = str(raw["problemId"])
            documents.append(
                BM25Document(
                    id=document_id,
                    text=text,
                    payload=document_payload,
                )
            )
        return cls(tuple(documents))

    def index_documents(self, documents: Sequence[BM25Document]) -> None:
        self._store.index_documents(documents)

    def search(self, query: str, *, top_k: int) -> tuple[SearchCandidate, ...]:
        return self._store.search(query, top_k=top_k)


def load_runtime_retrieval_settings(
    environ: Mapping[str, str] | None = None,
) -> RuntimeRetrievalSettings:
    values = os.environ if environ is None else environ
    backend = values.get("RETRIEVAL_BACKEND", "local").strip().lower()
    if backend not in {"local", "stores"}:
        raise RuntimeRetrievalError(f"unsupported RETRIEVAL_BACKEND: {backend}")

    bm25_path = _resolve_repo_path(values.get("BM25_INDEX_PATH", "data/processed/bm25_index.json"))
    return RuntimeRetrievalSettings(
        backend=backend,  # type: ignore[arg-type]
        qdrant_url=values.get("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=values.get("QDRANT_COLLECTION", "programming_chunks"),
        neo4j_uri=values.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=values.get("NEO4J_USER", "neo4j"),
        neo4j_password=values.get("NEO4J_PASSWORD", "password"),
        bm25_index_path=bm25_path,
    )


def build_runtime_retrieval(
    settings: RuntimeRetrievalSettings | None = None,
    *,
    documents: Sequence[RetrievalDocument] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> RuntimeRetrieval:
    resolved = settings or load_runtime_retrieval_settings()
    if resolved.backend == "local":
        return RuntimeRetrieval(
            backend="local",
            pipeline=OnlineQueryPipeline(
                documents=documents,
                embedding_provider=embedding_provider,
            ),
            candidate_sources={"vector": "local", "graph": "local", "bm25": "local"},
        )
    elif resolved.backend == "stores":
        # Qdrant and Neo4j constructors do not intentionally perform health checks here.
        # Connection problems may surface when the first query executes.
        vector_store = QdrantVectorStore(
            url=resolved.qdrant_url,
            collection_name=resolved.qdrant_collection,
        )
        graph_store = Neo4jGraphStore(
            uri=resolved.neo4j_uri,
            user=resolved.neo4j_user,
            password=resolved.neo4j_password,
        )
        bm25_store = JsonBM25Store.from_path(resolved.bm25_index_path)
        return RuntimeRetrieval(
            backend="stores",
            pipeline=OnlineQueryPipeline(
                documents=documents,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                graph_store=graph_store,
                bm25_store=bm25_store,
            ),
            candidate_sources={"vector": "qdrant", "graph": "neo4j", "bm25": "bm25_index"},
        )

    raise RuntimeRetrievalError(f"unsupported retrieval backend: {resolved.backend}")


def add_runtime_debug_trace(
    trace: JsonMap,
    candidate_sources: Mapping[str, str],
) -> JsonMap:
    labeled = dict(trace)
    source_labels = dict(candidate_sources)
    labeled["candidateSources"] = source_labels
    for key, source_key in (
        ("vectorCandidates", "vector"),
        ("graphCandidates", "graph"),
        ("bm25Candidates", "bm25"),
    ):
        labeled[key] = [
            {**dict(candidate), "candidateSource": source_labels[source_key]}
            for candidate in labeled.get(key, [])
        ]
    for key in ("fusionScores", "rerankerScores"):
        labeled[key] = [
            {**dict(candidate), "candidateSource": "hybrid"}
            for candidate in labeled.get(key, [])
        ]
    return labeled


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]

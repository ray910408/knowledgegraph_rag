from __future__ import annotations

import uuid
from typing import Any, Sequence

from ..stores import SearchCandidate, VectorRecord


class QdrantAdapterError(RuntimeError):
    pass


class QdrantVectorStore:
    def __init__(
        self,
        *,
        client: Any | None = None,
        collection_name: str = "programming_chunks",
        url: str = "http://localhost:6333",
        timeout: float = 1.0,
    ) -> None:
        self.collection_name = collection_name
        if client is not None:
            self._client = client
            return
        try:
            from qdrant_client import QdrantClient
        except Exception as exc:  # pragma: no cover - depends on optional install state
            raise QdrantAdapterError("qdrant-client is not installed") from exc
        self._client = QdrantClient(url=url, timeout=timeout)

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        self._ensure_collection(len(records[0].vector))
        points = [_to_qdrant_point(record) for record in records]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> tuple[SearchCandidate, ...]:
        if hasattr(self._client, "query_points"):
            result = self._client.query_points(
                collection_name=self.collection_name,
                query=list(query_vector),
                limit=top_k,
                query_filter=filters,
            )
            points = getattr(result, "points", result)
        else:
            points = self._client.search(
                collection_name=self.collection_name,
                query_vector=list(query_vector),
                limit=top_k,
                query_filter=filters,
            )
        return tuple(_candidate_from_point(point) for point in points)

    def _ensure_collection(self, vector_size: int) -> None:
        if not hasattr(self._client, "collection_exists"):
            return
        if self._client.collection_exists(self.collection_name):
            return
        try:
            from qdrant_client.http.models import Distance, VectorParams
        except Exception as exc:  # pragma: no cover - depends on optional install state
            raise QdrantAdapterError("qdrant-client models are not available") from exc
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _to_qdrant_point(record: VectorRecord) -> Any:
    payload = dict(record.payload)
    payload["_recordId"] = record.id
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, record.id))
    try:
        from qdrant_client.http.models import PointStruct
    except Exception:
        return {"id": point_id, "vector": list(record.vector), "payload": payload}
    return PointStruct(id=point_id, vector=list(record.vector), payload=payload)


def _candidate_from_point(point: Any) -> SearchCandidate:
    payload = dict(getattr(point, "payload", None) or {})
    point_id = str(payload.pop("_recordId", getattr(point, "id")))
    return SearchCandidate(
        id=point_id,
        score=float(getattr(point, "score", 0.0)),
        payload=payload,
    )

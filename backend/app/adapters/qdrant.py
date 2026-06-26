from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata
import uuid
from typing import Any, Sequence

from ..stores import SearchCandidate, VectorRecord


class QdrantAdapterError(RuntimeError):
    pass


def qdrant_compatibility_warning(client: Any) -> dict[str, str] | None:
    client_version_text = _client_version_text(client)
    if client_version_text is None:
        return None
    server_version_text = _server_version_text(client)
    if server_version_text is None:
        return None

    client_minor = _major_minor(client_version_text)
    server_minor = _major_minor(server_version_text)
    if client_minor is None or server_minor is None or client_minor == server_minor:
        return None
    return {
        "adapter": "qdrant",
        "severity": "warning",
        "message": (
            f"Qdrant client {client_version_text} is outside the supported server "
            f"{server_version_text} minor range."
        ),
    }


def _client_version_text(client: Any) -> str | None:
    missing = object()
    try:
        client_version = getattr(client, "client_version", missing)
    except Exception:
        return None
    if client_version is missing:
        try:
            client_version = getattr(client, "version", missing)
        except Exception:
            return None
    if client_version is missing:
        try:
            client_version = metadata.version("qdrant-client")
        except Exception:
            return None
    elif client_version is None:
        try:
            client_version = getattr(client, "version", missing)
        except Exception:
            return None
        if client_version is missing:
            return None
    return _version_text(client_version)


def _server_version_text(client: Any) -> str | None:
    try:
        info = getattr(client, "info", None)
    except Exception:
        return None
    if callable(info):
        try:
            return _version_from_info(info())
        except Exception:
            return None

    try:
        get_version = getattr(client, "get_version", None)
    except Exception:
        return None
    if callable(get_version):
        try:
            return _version_from_info(get_version())
        except Exception:
            return None
    return None


def _version_from_info(value: object) -> str | None:
    if isinstance(value, Mapping):
        return _version_text(value.get("version"))
    if isinstance(value, str):
        return _version_text(value)
    try:
        return _version_text(getattr(value, "version", None))
    except Exception:
        return None


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
        self._client = QdrantClient(
            url=url,
            timeout=timeout,
            check_compatibility=False,
        )

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

    def compatibility_warning(self) -> dict[str, str] | None:
        return qdrant_compatibility_warning(self._client)

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


def _version_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text or None


def _major_minor(version: str) -> str | None:
    parts = version.split(".")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return ".".join(parts[:2])

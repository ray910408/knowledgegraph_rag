from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    @property
    def provider_kind(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class DeterministicMockEmbeddingProvider:
    def __init__(self, *, dimension: int = 32, model_name: str = "BAAI/bge-m3") -> None:
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_kind(self) -> str:
        return "mock"

    def embed_text(self, text: str) -> list[float]:
        normalized = " ".join(text.lower().split())
        values: list[float] = []
        counter = 0
        while len(values) < self._dimension:
            digest = hashlib.sha256(f"{normalized}|{counter}".encode("utf-8")).digest()
            for byte in digest:
                values.append((byte / 127.5) - 1.0)
                if len(values) == self._dimension:
                    break
            counter += 1
        return _unit_normalize(values)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


def _unit_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]

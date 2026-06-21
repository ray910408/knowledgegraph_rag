from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalModelConfig:
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    language: str = "zh-Hant"


DEFAULT_RETRIEVAL_CONFIG = RetrievalModelConfig()

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class Problem:
    id: str
    title: str
    text: str
    concept_ids: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Concept:
    id: str
    name: str
    kind: str = "concept"
    description: str = ""
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Relationship:
    source_id: str
    target_id: str
    type: str
    weight: float = 1.0
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidencePath:
    nodes: tuple[str, ...]
    relations: tuple[str, ...]
    score: float
    rationale: str


@dataclass(frozen=True)
class Recommendation:
    problem_id: str
    title: str
    score: float
    vector_score: float
    graph_score: float
    concept_match_score: float
    concept_matches: tuple[str, ...] = ()
    evidence_paths: tuple[EvidencePath, ...] = ()


@dataclass(frozen=True)
class TechniqueRecommendation:
    id: str
    kind: str
    title: str
    score: float
    confidence: str
    summary: str
    fit_signals: tuple[str, ...] = ()
    pitfalls: tuple[str, ...] = ()
    evidence_paths: tuple[EvidencePath, ...] = ()


@dataclass(frozen=True)
class EvidenceBundle:
    query_problem: Problem
    recommendations: tuple[Recommendation, ...]


@dataclass(frozen=True)
class VectorCandidate:
    problem: Problem
    score: float

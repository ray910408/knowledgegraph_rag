from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JsonMap = dict[str, Any]


def _tuple_of_str(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value) for value in values)


def _tuple_of_mapping(values: Any) -> tuple[JsonMap, ...]:
    if values is None:
        return ()
    return tuple(dict(value) for value in values)


def _dict(value: Any) -> JsonMap:
    return dict(value or {})


@dataclass(frozen=True)
class RawProblem:
    id: str
    source: str
    source_id: str
    title: str
    problem_type: str
    statement: str
    answer: str
    solution_hints: tuple[str, ...] = ()
    concepts: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: JsonMap = field(default_factory=dict)
    difficulty: str | None = None
    constraints: tuple[str, ...] = ()
    examples: tuple[JsonMap, ...] = ()
    editorial: str | None = None

    @classmethod
    def from_mapping(cls, raw: JsonMap) -> RawProblem:
        return cls(
            id=str(raw["id"]),
            source=str(raw["source"]),
            source_id=str(raw["sourceId"]),
            title=str(raw["title"]),
            problem_type=str(raw["problemType"]),
            statement=str(raw["statement"]),
            answer=str(raw["answer"]),
            solution_hints=_tuple_of_str(raw.get("solutionHints")),
            concepts=_tuple_of_str(raw.get("concepts")),
            tags=_tuple_of_str(raw.get("tags")),
            metadata=_dict(raw.get("metadata")),
            difficulty=str(raw["difficulty"]) if raw.get("difficulty") is not None else None,
            constraints=_tuple_of_str(raw.get("constraints")),
            examples=_tuple_of_mapping(raw.get("examples")),
            editorial=str(raw["editorial"]) if raw.get("editorial") is not None else None,
        )

    def to_mapping(self) -> JsonMap:
        data: JsonMap = {
            "id": self.id,
            "source": self.source,
            "sourceId": self.source_id,
            "title": self.title,
            "problemType": self.problem_type,
            "statement": self.statement,
            "answer": self.answer,
            "solutionHints": list(self.solution_hints),
            "concepts": list(self.concepts),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }
        if self.difficulty is not None:
            data["difficulty"] = self.difficulty
        if self.constraints:
            data["constraints"] = list(self.constraints)
        if self.examples:
            data["examples"] = [dict(example) for example in self.examples]
        if self.editorial is not None:
            data["editorial"] = self.editorial
        return data


@dataclass(frozen=True)
class ProblemChunk:
    id: str
    problem_id: str
    kind: str
    text: str
    index: int
    chunk_id: str = ""
    input_id: str = ""
    display_text: str = ""
    search_text: str = ""
    concepts: tuple[str, ...] = ()
    metadata: JsonMap = field(default_factory=dict)
    answer: str = ""
    solution_hints: tuple[str, ...] = ()
    difficulty: str | None = None
    constraints: tuple[str, ...] = ()
    examples: tuple[JsonMap, ...] = ()
    editorial: str | None = None
    source: str = ""
    source_id: str = ""
    title: str = ""
    problem_type: str = ""

    def to_mapping(self) -> JsonMap:
        # Rollout 1 keeps the legacy text alias stable while additive fields land.
        display_text = self.display_text or self.text
        return {
            "id": self.id,
            "problemId": self.problem_id,
            "kind": self.kind,
            "text": display_text,
            "displayText": display_text,
            "searchText": self.search_text,
            "chunkId": self.chunk_id or self.id,
            "inputId": self.input_id,
            "index": self.index,
            "concepts": list(self.concepts),
            "metadata": dict(self.metadata),
            "answer": self.answer,
            "solutionHints": list(self.solution_hints),
            "difficulty": self.difficulty,
            "constraints": list(self.constraints),
            "examples": [dict(example) for example in self.examples],
            "editorial": self.editorial,
            "source": self.source,
            "sourceId": self.source_id,
            "title": self.title,
            "problemType": self.problem_type,
        }


@dataclass(frozen=True)
class EntityRecord:
    id: str
    name: str
    type: str
    aliases: tuple[str, ...] = ()
    problem_ids: tuple[str, ...] = ()
    metadata: JsonMap = field(default_factory=dict)

    def to_mapping(self) -> JsonMap:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "aliases": list(self.aliases),
            "problemIds": list(self.problem_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RelationRecord:
    id: str
    source_id: str
    target_id: str
    type: str
    weight: float = 1.0
    evidence: tuple[str, ...] = ()
    metadata: JsonMap = field(default_factory=dict)

    def to_mapping(self) -> JsonMap:
        return {
            "id": self.id,
            "sourceId": self.source_id,
            "targetId": self.target_id,
            "type": self.type,
            "weight": self.weight,
            "evidence": list(self.evidence),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ScoreMetadata:
    stage: str
    display_label: str
    comparable_across_stages: bool = False

    def to_mapping(self) -> JsonMap:
        return {
            "stage": self.stage,
            "displayLabel": self.display_label,
            "comparableAcrossStages": self.comparable_across_stages,
        }


@dataclass(frozen=True)
class RetrievalTrace:
    query_understanding: JsonMap = field(default_factory=dict)
    entity_linking: list[JsonMap] = field(default_factory=list)
    vector_candidates: list[JsonMap] = field(default_factory=list)
    graph_candidates: list[JsonMap] = field(default_factory=list)
    bm25_candidates: list[JsonMap] = field(default_factory=list)
    fusion_scores: list[JsonMap] = field(default_factory=list)
    reranker_scores: list[JsonMap] = field(default_factory=list)
    graph_search_status: str = field(default="none", kw_only=True)
    matched_problem: JsonMap | None = None

    def to_mapping(self) -> JsonMap:
        return {
            "queryUnderstanding": dict(self.query_understanding),
            "entityLinking": [dict(item) for item in self.entity_linking],
            "vectorCandidates": [dict(item) for item in self.vector_candidates],
            "graphCandidates": [dict(item) for item in self.graph_candidates],
            "graphSearchStatus": self.graph_search_status,
            "bm25Candidates": [dict(item) for item in self.bm25_candidates],
            "fusionScores": [dict(item) for item in self.fusion_scores],
            "rerankerScores": [dict(item) for item in self.reranker_scores],
            "matchedProblem": dict(self.matched_problem) if self.matched_problem is not None else None,
        }


@dataclass(frozen=True)
class RetrievalEvidenceBundle:
    similar_problems: list[JsonMap] = field(default_factory=list)
    graph_paths: list[JsonMap] = field(default_factory=list)
    algorithm_evidence: list[str] = field(default_factory=list)
    data_structure_evidence: list[str] = field(default_factory=list)
    pattern_evidence: list[str] = field(default_factory=list)
    technique_evidence: list[str] = field(default_factory=list)
    common_mistakes: list[str] = field(default_factory=list)
    matched_problem: JsonMap | None = None

    def to_mapping(self) -> JsonMap:
        return {
            "similarProblems": [dict(item) for item in self.similar_problems],
            "graphPaths": [dict(item) for item in self.graph_paths],
            "algorithmEvidence": list(self.algorithm_evidence),
            "dataStructureEvidence": list(self.data_structure_evidence),
            "patternEvidence": list(self.pattern_evidence),
            "techniqueEvidence": list(self.technique_evidence),
            "commonMistakes": list(self.common_mistakes),
            "matchedProblem": dict(self.matched_problem) if self.matched_problem is not None else None,
        }

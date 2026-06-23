from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from ..analysis import detect_input_kind, load_programming_dataset
from ..contracts import RetrievalEvidenceBundle, RetrievalTrace
from ..providers import DeterministicMockEmbeddingProvider, EmbeddingProvider
from ..stores import BM25Store, GraphStore, SearchCandidate, VectorStore


JsonMap = dict[str, Any]


@dataclass(frozen=True)
class RetrievalDocument:
    id: str
    source: str
    source_id: str
    title: str
    text: str
    answer: str
    concepts: tuple[str, ...]
    problem_type: str
    solution_hints: tuple[str, ...] = ()
    difficulty: str = ""
    constraints: tuple[str, ...] = ()
    examples: tuple[JsonMap, ...] = ()
    editorial: str = ""
    metadata: JsonMap = field(default_factory=dict)


@dataclass(frozen=True)
class QueryUnderstanding:
    original_query: str
    normalized_query: str
    input_kind: str
    intent: str
    keywords: tuple[str, ...]

    def to_mapping(self) -> JsonMap:
        return {
            "originalQuery": self.original_query,
            "normalizedQuery": self.normalized_query,
            "inputKind": self.input_kind,
            "intent": self.intent,
            "keywords": list(self.keywords),
        }


@dataclass(frozen=True)
class RetrievalCandidate:
    id: str
    title: str
    source: str
    score: float
    text: str = ""
    concepts: tuple[str, ...] = ()
    problem_type: str = ""
    payload: JsonMap = field(default_factory=dict)

    def to_mapping(self) -> JsonMap:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "score": round(self.score, 6),
            "concepts": list(self.concepts),
            "problemType": self.problem_type,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class GraphSearchResult:
    candidates: tuple[RetrievalCandidate, ...]
    paths: tuple[JsonMap, ...]


@dataclass(frozen=True)
class OnlineQueryResult:
    query_understanding: QueryUnderstanding
    linked_entities: tuple[JsonMap, ...]
    vector_candidates: tuple[RetrievalCandidate, ...]
    graph_candidates: tuple[RetrievalCandidate, ...]
    bm25_candidates: tuple[RetrievalCandidate, ...]
    fused_candidates: tuple[RetrievalCandidate, ...]
    reranked_candidates: tuple[RetrievalCandidate, ...]
    graph_paths: tuple[JsonMap, ...]
    trace: RetrievalTrace


class QueryUnderstandingService:
    def understand(self, query: str) -> QueryUnderstanding:
        normalized = " ".join(query.strip().split())
        lowered = normalized.lower()
        keywords = tuple(dict.fromkeys(_tokens(lowered)))
        input_kind = detect_input_kind(query)
        intent = "code_analysis" if input_kind in {"cpp", "python"} else "problem_search"
        return QueryUnderstanding(
            original_query=query,
            normalized_query=normalized,
            input_kind=input_kind,
            intent=intent,
            keywords=keywords,
        )


class EntityLinkingService:
    _concept_aliases: tuple[tuple[str, str, str], ...] = (
        ("concept:bfs", "BFS", "algorithm"),
        ("concept:queue", "Queue", "data_structure"),
        ("concept:visited-array", "Visited Array", "data_structure"),
        ("concept:dynamic-programming", "Dynamic Programming", "algorithm"),
        ("pattern:graph-traversal", "Graph Traversal", "pattern"),
        ("concept:shortest-path", "Shortest Path", "concept"),
    )

    def link(self, understanding: QueryUnderstanding) -> tuple[JsonMap, ...]:
        text = understanding.normalized_query.lower()
        linked: list[JsonMap] = []
        for entity_id, name, kind in self._concept_aliases:
            terms = _tokens(name)
            if any(term in understanding.keywords for term in terms) or name.lower() in text:
                linked.append(
                    {
                        "entityId": entity_id,
                        "name": name,
                        "type": kind,
                        "confidence": 1.0,
                    }
                )
        return tuple(linked)


class VectorSearchService:
    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._documents = tuple(documents)
        self._embedding_provider = embedding_provider or DeterministicMockEmbeddingProvider()
        self._vector_store = vector_store
        self._vectors = {} if vector_store is not None else {
            document.id: self._embedding_provider.embed_text(
                f"{document.title} {document.text} {' '.join(document.concepts)}"
            )
            for document in self._documents
        }

    def search(self, understanding: QueryUnderstanding, *, top_k: int) -> tuple[RetrievalCandidate, ...]:
        query_vector = self._embedding_provider.embed_text(understanding.normalized_query)
        if self._vector_store is not None:
            return tuple(
                _candidate_from_store_candidate(candidate, source="vector")
                for candidate in self._vector_store.search(query_vector, top_k=top_k)
            )

        candidates = [
            _candidate_from_document(
                document,
                source="vector",
                score=_cosine(query_vector, self._vectors[document.id]),
            )
            for document in self._documents
        ]
        return tuple(sorted(candidates, key=lambda item: (-item.score, item.id))[:top_k])


class BM25SearchService:
    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        bm25_store: BM25Store | None = None,
    ) -> None:
        self._documents = tuple(documents)
        self._bm25_store = bm25_store

    def search(self, understanding: QueryUnderstanding, *, top_k: int) -> tuple[RetrievalCandidate, ...]:
        if self._bm25_store is not None:
            return tuple(
                _candidate_from_store_candidate(candidate, source="bm25")
                for candidate in self._bm25_store.search(understanding.normalized_query, top_k=top_k)
            )

        query_terms = set(understanding.keywords)
        candidates: list[RetrievalCandidate] = []
        for document in self._documents:
            terms = _tokens(
                f"{document.title} {document.text} {document.answer} {' '.join(document.concepts)}"
            )
            if not query_terms:
                score = 0.0
            else:
                score = sum(1 for term in terms if term in query_terms) / max(len(terms), 1)
                score += len(query_terms & set(terms)) / len(query_terms)
            candidates.append(_candidate_from_document(document, source="bm25", score=score))
        return tuple(sorted(candidates, key=lambda item: (-item.score, item.id))[:top_k])


class GraphSearchService:
    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        graph_store: GraphStore | None = None,
    ) -> None:
        self._documents = tuple(documents)
        self._graph_store = graph_store

    def search(self, linked_entities: Sequence[JsonMap], *, top_k: int) -> GraphSearchResult:
        if self._graph_store is not None:
            return self._search_store(linked_entities, top_k=top_k)

        entity_names = {str(entity["name"]).lower() for entity in linked_entities}
        entity_ids = {str(entity["entityId"]) for entity in linked_entities}
        candidates: list[RetrievalCandidate] = []
        paths: list[JsonMap] = []
        for document in self._documents:
            matched: list[JsonMap] = []
            for concept in document.concepts:
                concept_id = f"concept:{_slug(concept)}"
                if concept.lower() in entity_names or concept_id in entity_ids:
                    matched.append({"entityId": concept_id, "name": concept})
            pattern_id = f"pattern:{_slug(document.problem_type)}"
            if document.problem_type.lower() in entity_names or pattern_id in entity_ids:
                matched.append({"entityId": pattern_id, "name": document.problem_type})
            if not matched:
                continue
            score = len(matched) / max(len(linked_entities), 1)
            candidates.append(_candidate_from_document(document, source="graph", score=score))
            for match in matched:
                paths.append(
                    {
                        "nodes": ["input", match["entityId"], document.id],
                        "relations": ["MENTIONS", "REQUIRED_BY"],
                        "score": round(score, 6),
                        "rationale": f"linked {match['name']} to {document.title}",
                    }
                )
        return GraphSearchResult(
            candidates=tuple(sorted(candidates, key=lambda item: (-item.score, item.id))[:top_k]),
            paths=tuple(paths),
        )

    def _search_store(self, linked_entities: Sequence[JsonMap], *, top_k: int) -> GraphSearchResult:
        assert self._graph_store is not None
        candidates: list[RetrievalCandidate] = []
        paths: list[JsonMap] = []
        linked_entity_count = max(len(linked_entities), 1)
        for document in self._documents:
            document_paths_by_entity: dict[str, list[JsonMap]] = {}
            for entity in linked_entities:
                entity_id = str(entity["entityId"])
                direct_paths = self._graph_store.find_paths(document.id, entity_id, max_hops=3)
                reverse_paths = self._graph_store.find_paths(entity_id, document.id, max_hops=3)
                for path in (*direct_paths, *reverse_paths):
                    normalized = _normalize_graph_store_path(
                        path,
                        entity=entity,
                        document=document,
                    )
                    document_paths_by_entity.setdefault(entity_id, []).append(normalized)
                    paths.append(normalized)

            if not document_paths_by_entity:
                continue
            per_entity_scores = [
                max(float(path.get("score", 0.0)) for path in entity_paths)
                for entity_paths in document_paths_by_entity.values()
            ]
            score = min(sum(per_entity_scores) / linked_entity_count, 1.0)
            candidates.append(_candidate_from_document(document, source="graph", score=score))

        return GraphSearchResult(
            candidates=tuple(sorted(candidates, key=lambda item: (-item.score, item.id))[:top_k]),
            paths=tuple(paths),
        )


class HybridFusionService:
    _weights = {"vector": 0.35, "graph": 0.35, "bm25": 0.30}
    _source_order = {"graph": 0, "vector": 1, "bm25": 2}

    def fuse(
        self,
        *,
        vector_candidates: Sequence[RetrievalCandidate],
        graph_candidates: Sequence[RetrievalCandidate],
        bm25_candidates: Sequence[RetrievalCandidate],
        top_k: int,
    ) -> tuple[RetrievalCandidate, ...]:
        groups = {
            "vector": tuple(vector_candidates),
            "graph": tuple(graph_candidates),
            "bm25": tuple(bm25_candidates),
        }
        by_id: dict[str, RetrievalCandidate] = {}
        scores: dict[str, float] = defaultdict_float()
        sources: dict[str, set[str]] = {}
        for source, candidates in groups.items():
            best_by_id: dict[str, RetrievalCandidate] = {}
            for candidate in candidates:
                current = best_by_id.get(candidate.id)
                if current is None or candidate.score > current.score:
                    best_by_id[candidate.id] = candidate
            max_score = max((candidate.score for candidate in best_by_id.values()), default=0.0) or 1.0
            for candidate in best_by_id.values():
                by_id.setdefault(candidate.id, candidate)
                scores[candidate.id] = scores.get(candidate.id, 0.0) + (
                    self._weights[source] * (candidate.score / max_score)
                )
                sources.setdefault(candidate.id, set()).add(source)

        fused: list[RetrievalCandidate] = []
        for candidate_id, candidate in by_id.items():
            ordered_sources = sorted(sources[candidate_id], key=self._source_order.get)
            payload = dict(candidate.payload)
            payload["sources"] = ordered_sources
            fused.append(
                replace(
                    candidate,
                    source="hybrid",
                    score=round(min(scores[candidate_id], 1.0), 6),
                    payload=payload,
                )
            )
        return tuple(sorted(fused, key=lambda item: (-item.score, item.id))[:top_k])


class Reranker:
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_k: int,
    ) -> tuple[RetrievalCandidate, ...]:
        query_terms = set(_tokens(query))
        reranked: list[RetrievalCandidate] = []
        for candidate in candidates:
            candidate_terms = set(
                _tokens(f"{candidate.title} {candidate.text} {' '.join(candidate.concepts)}")
            )
            lexical = len(query_terms & candidate_terms) / max(len(query_terms), 1)
            reranker_score = round((0.7 * lexical) + (0.3 * candidate.score), 6)
            payload = dict(candidate.payload)
            payload["rerankerScore"] = reranker_score
            reranked.append(replace(candidate, score=reranker_score, payload=payload))
        return tuple(sorted(reranked, key=lambda item: (-item.score, item.id))[:top_k])


class EvidenceBuilder:
    def build(
        self,
        candidates: Sequence[RetrievalCandidate],
        graph_paths: Sequence[JsonMap],
    ) -> RetrievalEvidenceBundle:
        algorithms: list[str] = []
        data_structures: list[str] = []
        patterns: list[str] = []
        for candidate in candidates:
            for concept in candidate.concepts:
                kind = _classify_concept(concept)
                if kind == "algorithm":
                    _append_unique(algorithms, concept)
                elif kind == "data_structure":
                    _append_unique(data_structures, concept)
            if candidate.problem_type:
                _append_unique(patterns, candidate.problem_type)

        return RetrievalEvidenceBundle(
            similar_problems=[
                {
                    "id": candidate.id,
                    "title": candidate.title,
                    "score": round(candidate.score, 6),
                    "sharedConcepts": list(candidate.concepts),
                    "answerHint": candidate.payload.get("answer", ""),
                }
                for candidate in candidates
            ],
            graph_paths=[dict(path) for path in graph_paths],
            algorithm_evidence=algorithms,
            data_structure_evidence=data_structures,
            pattern_evidence=patterns,
            common_mistakes=[
                "forget visited state and revisit the same node",
                "enqueue a node before recording its distance or state",
            ],
        )


class ContextBuilder:
    def build(
        self,
        query_understanding: QueryUnderstanding,
        evidence_bundle: RetrievalEvidenceBundle,
    ) -> str:
        evidence = evidence_bundle.to_mapping()
        lines = [
            "Query Understanding",
            f"- intent: {query_understanding.intent}",
            f"- inputKind: {query_understanding.input_kind}",
            f"- keywords: {', '.join(query_understanding.keywords)}",
            "",
            "Similar Problems",
        ]
        for problem in evidence["similarProblems"]:
            lines.append(
                f"- {problem['id']} {problem['title']} "
                f"(score={problem['score']}, concepts={', '.join(problem['sharedConcepts'])})"
            )
        lines.extend(
            [
                "",
                "Graph Paths",
                *[f"- {' -> '.join(path.get('nodes', []))}" for path in evidence["graphPaths"]],
                "",
                "Algorithm Evidence",
                f"- {', '.join(evidence['algorithmEvidence'])}",
                "Data Structure Evidence",
                f"- {', '.join(evidence['dataStructureEvidence'])}",
                "Pattern Evidence",
                f"- {', '.join(evidence['patternEvidence'])}",
                "Common Mistakes",
                *[f"- {mistake}" for mistake in evidence["commonMistakes"]],
            ]
        )
        return "\n".join(lines)


class LLMResponseGenerator:
    def generate(self, context: str) -> str:
        return "Evidence-backed response generated from context.\n\n" + context


class OnlineQueryPipeline:
    def __init__(
        self,
        *,
        documents: Sequence[RetrievalDocument] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        self._documents = tuple(documents) if documents is not None else _load_default_documents()
        self._embedding_provider = embedding_provider or DeterministicMockEmbeddingProvider()
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._graph_store = graph_store

    def run(self, query: str, *, top_k: int = 5) -> OnlineQueryResult:
        understanding = QueryUnderstandingService().understand(query)
        linked_entities = EntityLinkingService().link(understanding)
        vector_candidates = VectorSearchService(
            self._documents,
            self._embedding_provider,
            vector_store=self._vector_store,
        ).search(understanding, top_k=max(top_k * 2, top_k))
        bm25_candidates = BM25SearchService(
            self._documents,
            bm25_store=self._bm25_store,
        ).search(
            understanding,
            top_k=max(top_k * 2, top_k),
        )
        graph_result = GraphSearchService(
            self._documents,
            graph_store=self._graph_store,
        ).search(
            linked_entities,
            top_k=max(top_k * 2, top_k),
        )
        fused = HybridFusionService().fuse(
            vector_candidates=vector_candidates,
            graph_candidates=graph_result.candidates,
            bm25_candidates=bm25_candidates,
            top_k=max(top_k * 2, top_k),
        )
        reranked = Reranker().rerank(understanding.normalized_query, fused, top_k=top_k)
        trace = RetrievalTrace(
            query_understanding=understanding.to_mapping(),
            entity_linking=[dict(entity) for entity in linked_entities],
            vector_candidates=[candidate.to_mapping() for candidate in vector_candidates],
            graph_candidates=[candidate.to_mapping() for candidate in graph_result.candidates],
            bm25_candidates=[candidate.to_mapping() for candidate in bm25_candidates],
            fusion_scores=[candidate.to_mapping() for candidate in fused],
            reranker_scores=[candidate.to_mapping() for candidate in reranked],
        )
        return OnlineQueryResult(
            query_understanding=understanding,
            linked_entities=linked_entities,
            vector_candidates=vector_candidates,
            graph_candidates=graph_result.candidates,
            bm25_candidates=bm25_candidates,
            fused_candidates=fused,
            reranked_candidates=reranked,
            graph_paths=graph_result.paths,
            trace=trace,
        )


def _candidate_from_document(
    document: RetrievalDocument,
    *,
    source: str,
    score: float,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        id=document.id,
        title=document.title,
        source=source,
        score=round(score, 6),
        text=document.text,
        concepts=document.concepts,
        problem_type=document.problem_type,
        payload={
            "documentSource": document.source,
            "sourceId": document.source_id,
            "answer": document.answer,
        },
    )


def _candidate_from_store_candidate(
    candidate: SearchCandidate,
    *,
    source: str,
) -> RetrievalCandidate:
    payload = dict(candidate.payload)
    metadata = _mapping(payload.get("metadata"))
    problem_id = str(
        payload.get("problemId")
        or payload.get("problem_id")
        or metadata.get("problemId")
        or candidate.id
    )
    concepts = _tuple_of_str(payload.get("concepts") or metadata.get("concepts"))
    title = str(payload.get("title") or metadata.get("title") or problem_id)
    problem_type = str(payload.get("problemType") or metadata.get("problemType") or "")
    text = str(payload.get("text") or payload.get("statement") or "")
    answer = str(payload.get("answer") or metadata.get("answer") or "")

    return RetrievalCandidate(
        id=problem_id,
        title=title,
        source=source,
        score=round(candidate.score, 6),
        text=text,
        concepts=concepts,
        problem_type=problem_type,
        payload={
            "storeCandidateId": candidate.id,
            "documentSource": str(payload.get("source") or metadata.get("source") or ""),
            "sourceId": str(payload.get("sourceId") or metadata.get("sourceId") or ""),
            "answer": answer,
            "storePayload": payload,
        },
    )


def _normalize_graph_store_path(
    path: JsonMap,
    *,
    entity: JsonMap,
    document: RetrievalDocument,
) -> JsonMap:
    entity_id = str(entity["entityId"])
    entity_name = str(entity.get("name", entity_id))
    raw_nodes = [str(node) for node in path.get("nodes", [])]
    raw_relations = [str(relation) for relation in path.get("relations", [])]

    return {
        "nodes": ["input", entity_id, document.id],
        "relations": ["MENTIONS", "REQUIRED_BY"],
        "score": round(float(path.get("score", 0.0)), 6),
        "rationale": f"linked {entity_name} to {document.title}",
        "storePath": {
            "nodes": raw_nodes,
            "relations": raw_relations,
        },
    }


def _mapping(value: Any) -> JsonMap:
    return dict(value) if isinstance(value, dict) else {}


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return (str(value),)


def _load_default_documents() -> tuple[RetrievalDocument, ...]:
    return tuple(
        RetrievalDocument(
            id=problem.id,
            source=problem.source,
            source_id=problem.source_id,
            title=problem.title,
            text=problem.statement,
            answer=problem.answer,
            concepts=problem.concepts,
            problem_type=problem.problem_type,
            solution_hints=problem.solution_hints,
            difficulty=problem.metadata.get("difficulty", ""),
            metadata=problem.metadata,
        )
        for problem in load_programming_dataset()
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return (numerator / (left_norm * right_norm) + 1.0) / 2.0


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.lower()))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def _classify_concept(name: str) -> str:
    lowered = name.lower()
    if lowered in {"bfs", "dfs", "dijkstra", "dynamic programming", "binary search"}:
        return "algorithm"
    if lowered in {"queue", "stack", "heap", "visited array", "array", "hash map"}:
        return "data_structure"
    return "concept"


def _append_unique(values: list[str], item: str) -> None:
    if item not in values:
        values.append(item)


def defaultdict_float() -> dict[str, float]:
    return {}

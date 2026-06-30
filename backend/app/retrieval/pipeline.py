from __future__ import annotations

from copy import deepcopy
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Sequence

from ..analysis import detect_input_kind, load_programming_dataset
from ..contracts import RetrievalEvidenceBundle, RetrievalTrace, ScoreMetadata
from ..providers import DeterministicMockEmbeddingProvider, EmbeddingProvider
from ..query_language import (
    GRAPH_SEED_ENTITY_REGISTRY,
    build_query_language_profile,
    shared_multilingual_tokens,
)
from ..stores import (
    BM25Store,
    GraphQueryStore,
    GraphRelatedProblemLookup,
    SearchCandidate,
    VectorStore,
)


JsonMap = dict[str, Any]
RetrievalMode = Literal["hybrid", "vector", "graph"]
GraphSearchStatus = Literal["none", "candidates", "paths_only"]
_MAX_STORE_FETCH_ATTEMPTS = 4
_MAX_STORE_FETCH_WINDOW = 100
_FALLBACK_COMMON_MISTAKES = (
    "忘記標記 visited。",
    "queue 初始化錯誤，導致起點或距離沒有被正確設定。",
)
_GRAPH_PATH_SCORING_STRATEGY = "weighted_layered_path_v1"
_GRAPH_PATH_CONCEPT_PRIORITY = {
    "shortest path": 0,
    "bfs": 1,
    "queue": 2,
    "graph traversal": 3,
    "visited array": 4,
}
_ALLOWED_GRAPH_NODE_LAYERS = frozenset(
    {"problem", "chunk", "concept", "code_feature", "pattern", "source"}
)
_ALLOWED_GRAPH_RELATION_TYPES = frozenset(
    {
        "HAS_SECTION",
        "DERIVED_FROM_SOURCE",
        "MENTIONS_CONCEPT",
        "HAS_PATTERN",
        "REQUIRES",
        "HAS_CODE_FEATURE",
        "USES_DATA_STRUCTURE",
        "IMPLEMENTS_PATTERN",
        "SIMILAR_BY_FEATURE",
        "EXPANDED_FROM_EXACT_MATCH",
    }
)
_CODE_FEATURE_ORDER = ("bfs", "queue_frontier", "visited_state", "grid_traversal")
_CODE_FEATURE_TOKEN_MARKERS = frozenset(
    {
        "bfs",
        "breadth",
        "queue",
        "deque",
        "popleft",
        "frontier",
        "visited",
        "vis",
        "seen",
        "set",
        "grid",
        "matrix",
        "directions",
        "neighbors",
    }
)
_CODE_FEATURE_MARKERS = {
    "bfs": ("bfs", "breadth"),
    "queue_frontier": ("queue", "deque", "frontier", "popleft", ".front()", "q.front"),
    "visited_state": ("visited", "vis", "seen"),
    "grid_traversal": (
        "grid",
        "matrix",
        "vector<vector",
        "len(grid)",
        "directions",
        "neighbors",
    ),
}
_CODE_FEATURE_ENTITY_MAP = {
    "bfs": {
        "entityId": "concept:bfs",
        "name": "BFS",
        "type": "algorithm",
    },
    "queue_frontier": {
        "entityId": "concept:queue",
        "name": "Queue",
        "type": "data_structure",
    },
    "visited_state": {
        "entityId": "concept:visited-array",
        "name": "Visited Array",
        "type": "technique",
    },
    "grid_traversal": {
        "entityId": "pattern:graph-traversal",
        "name": "Grid Traversal",
        "type": "pattern",
    },
}
_CODE_TRAVERSAL_CONTEXT_MARKERS = (
    "bfs",
    "breadth",
    "shortest",
    "grid",
    "matrix",
    "vector<vector",
    "len(grid)",
    "directions",
    "neighbors",
)
_CODE_FEATURE_COMPATIBILITY_TERMS = {
    "bfs": ("bfs", "breadth first", "breadth-first"),
    "queue_frontier": ("queue", "deque", "frontier"),
    "visited_state": ("visited array", "visited set", "state tracking", "visited"),
    "grid_traversal": ("grid traversal", "graph traversal", "grid", "matrix"),
}


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
class CodeFeatures:
    language: str
    features: tuple[str, ...]

    def to_mapping(self) -> JsonMap:
        return {
            "language": self.language,
            "features": list(self.features),
        }


@dataclass(frozen=True)
class QueryUnderstanding:
    original_query: str
    normalized_query: str
    input_kind: str
    intent: str
    keywords: tuple[str, ...]
    query_language: str = "en"
    exact_terms: tuple[str, ...] = ()
    low_weight_terms: tuple[str, ...] = ()
    concept_seeds: tuple[str, ...] = ()
    expanded_terms: tuple[str, ...] = ()
    query_variants: JsonMap = field(default_factory=dict)
    code_features: CodeFeatures | None = None

    def to_mapping(self) -> JsonMap:
        mapping = {
            "originalQuery": self.original_query,
            "normalizedQuery": self.normalized_query,
            "inputKind": self.input_kind,
            "intent": self.intent,
            "keywords": list(self.keywords),
            "queryLanguage": self.query_language,
            "exactTerms": list(self.exact_terms),
            "lowWeightTerms": list(self.low_weight_terms),
            "conceptSeeds": list(self.concept_seeds),
            "expandedTerms": list(self.expanded_terms),
            "queryVariants": dict(self.query_variants),
        }
        if self.code_features is not None:
            mapping["codeFeatures"] = self.code_features.to_mapping()
        return mapping


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


ExactMatchKind = Literal["exact_problem_id", "exact_source_id", "exact_title", "partial_title"]


@dataclass(frozen=True)
class ExactProblemMatch:
    problem_id: str
    title: str
    source: str
    source_id: str
    match_kind: ExactMatchKind
    confidence: float
    candidate: RetrievalCandidate

    def to_mapping(self) -> JsonMap:
        return {
            "id": self.problem_id,
            "title": self.title,
            "source": self.source,
            "sourceId": self.source_id,
            "matchKind": self.match_kind,
            "confidence": round(self.confidence, 6),
            "score": round(self.candidate.score, 6),
            "sharedConcepts": list(self.candidate.concepts),
            "problemType": self.candidate.problem_type,
            "answerHint": self.candidate.payload.get("answer", ""),
            "solutionHints": list(_tuple_of_str(self.candidate.payload.get("solutionHints"))),
            "difficulty": str(self.candidate.payload.get("difficulty") or ""),
            "constraints": list(_tuple_of_str(self.candidate.payload.get("constraints"))),
        }


@dataclass(frozen=True)
class GraphSearchResult:
    candidates: tuple[RetrievalCandidate, ...]
    paths: tuple[JsonMap, ...]


@dataclass(frozen=True)
class OnlineQueryResult:
    query_understanding: QueryUnderstanding
    linked_entities: tuple[JsonMap, ...]
    matched_problem: ExactProblemMatch | None
    vector_candidates: tuple[RetrievalCandidate, ...]
    graph_candidates: tuple[RetrievalCandidate, ...]
    bm25_candidates: tuple[RetrievalCandidate, ...]
    fused_candidates: tuple[RetrievalCandidate, ...]
    reranked_candidates: tuple[RetrievalCandidate, ...]
    graph_paths: tuple[JsonMap, ...]
    trace: RetrievalTrace
    raw_graph_paths: tuple[JsonMap, ...] = ()


class QueryUnderstandingService:
    def __init__(self, documents: Sequence[RetrievalDocument] = ()) -> None:
        self._documents = tuple(documents)

    def understand(self, query: str) -> QueryUnderstanding:
        normalized = " ".join(query.strip().split())
        language_profile = build_query_language_profile(normalized)
        keywords = language_profile.keywords
        input_kind = detect_input_kind(query)
        exact_match = ExactProblemMatcher(self._documents).match(
            QueryUnderstanding(
                original_query=query,
                normalized_query=normalized,
                input_kind=input_kind,
                intent="problem_search",
                keywords=keywords,
            )
        )
        if exact_match and (exact_match.match_kind != "partial_title" or input_kind not in {"cpp", "python"}):
            input_kind = "problem"
        intent = "code_analysis" if input_kind in {"cpp", "python"} else "problem_search"
        code_features: CodeFeatures | None = None
        if input_kind in {"cpp", "python"}:
            code_features = CodeFeatureExtractor().extract(query, input_kind=input_kind)
            feature_keywords = tuple(
                feature.replace("_", " ")
                for feature in code_features.features
            )
            keywords = tuple(dict.fromkeys((*keywords, *feature_keywords)))
        return QueryUnderstanding(
            original_query=query,
            normalized_query=normalized,
            input_kind=input_kind,
            intent=intent,
            keywords=keywords,
            query_language=language_profile.query_language,
            exact_terms=language_profile.exact_terms,
            low_weight_terms=language_profile.low_weight_terms,
            concept_seeds=language_profile.concept_seeds,
            expanded_terms=language_profile.expanded_terms,
            query_variants={
                "bm25": language_profile.bm25_query,
                "vector": language_profile.vector_query,
                "graphSeeds": list(language_profile.graph_seeds),
            },
            code_features=code_features,
        )


class CodeFeatureExtractor:
    def extract(
        self,
        snippet: str,
        *,
        input_kind: str | None = None,
        language: str | None = None,
    ) -> CodeFeatures:
        resolved_language = input_kind or language or "unknown"
        lowered = snippet.lower()
        tokens = set(_tokens(lowered))
        has_traversal_context = self._has_any_marker(
            lowered,
            tokens,
            _CODE_TRAVERSAL_CONTEXT_MARKERS,
        )
        if not has_traversal_context:
            return CodeFeatures(language=resolved_language, features=())

        detected = {
            feature
            for feature in _CODE_FEATURE_ORDER
            if self._has_any_marker(lowered, tokens, _CODE_FEATURE_MARKERS[feature])
        }
        has_frontier = "queue_frontier" in detected
        if has_frontier and (
            "bfs" in detected
            or self._has_any_marker(lowered, tokens, ("shortest", "grid", "matrix", "vector<vector"))
        ):
            detected.add("bfs")
        elif "bfs" not in detected:
            detected.discard("queue_frontier")
        if "visited_state" in detected and not (has_frontier or "bfs" in detected):
            detected.discard("visited_state")
        features = tuple(feature for feature in _CODE_FEATURE_ORDER if feature in detected)
        return CodeFeatures(language=resolved_language, features=features)

    @staticmethod
    def _has_any_marker(
        lowered: str,
        tokens: set[str],
        markers: Sequence[str],
    ) -> bool:
        for marker in markers:
            if marker in _CODE_FEATURE_TOKEN_MARKERS:
                if marker in tokens:
                    return True
                continue
            if marker in lowered:
                return True
        return False


class ExactProblemMatcher:
    def __init__(self, documents: Sequence[RetrievalDocument]) -> None:
        self._documents = tuple(documents)

    def match(self, understanding: QueryUnderstanding) -> ExactProblemMatch | None:
        query = _normalize_alias(understanding.normalized_query)
        if not query:
            return None

        best: ExactProblemMatch | None = None
        for document in self._documents:
            candidate = _candidate_from_document(document, source="exact", score=1.0)
            aliases = _problem_aliases(document)
            exact_problem_ids = {
                _normalize_alias(document.id),
                _normalize_alias(document.id.replace("-", " ")),
                _normalize_alias(f"{document.source}-{document.source_id} {document.title}"),
                _normalize_alias(f"{document.source} {document.source_id} {document.title}"),
                _normalize_alias(f"{document.id} {document.title}"),
            }
            exact_source_ids = {
                _normalize_alias(document.source_id),
                _normalize_alias(f"{document.source}-{document.source_id}"),
                _normalize_alias(f"{document.source} {document.source_id}"),
            }
            exact_title = _normalize_alias(document.title)
            title_tokens = set(_tokens(document.title))
            query_tokens = set(understanding.keywords)

            if query in exact_problem_ids:
                match = ExactProblemMatch(
                    document.id,
                    document.title,
                    document.source,
                    document.source_id,
                    "exact_problem_id",
                    1.0,
                    candidate,
                )
            elif query in exact_source_ids:
                match = ExactProblemMatch(
                    document.id,
                    document.title,
                    document.source,
                    document.source_id,
                    "exact_source_id",
                    0.98,
                    candidate,
                )
            elif query == exact_title or query in aliases:
                match = ExactProblemMatch(
                    document.id,
                    document.title,
                    document.source,
                    document.source_id,
                    "exact_title",
                    0.96,
                    candidate,
                )
            elif title_tokens and len(title_tokens & query_tokens) >= min(2, len(title_tokens)):
                confidence = len(title_tokens & query_tokens) / len(title_tokens)
                match = ExactProblemMatch(
                    document.id,
                    document.title,
                    document.source,
                    document.source_id,
                    "partial_title",
                    round(0.70 + (0.20 * confidence), 6),
                    candidate,
                )
            else:
                continue

            if best is None or match.confidence > best.confidence:
                best = match
        return best


class EntityLinkingService:
    _concept_aliases: tuple[tuple[str, str, str], ...] = (
        *(
            (
                entity_id,
                str(metadata["name"]),
                str(metadata["type"]),
            )
            for entity_id, metadata in GRAPH_SEED_ENTITY_REGISTRY.items()
        ),
        ("concept:dynamic-programming", "Dynamic Programming", "algorithm"),
    )

    def link(
        self,
        understanding: QueryUnderstanding,
        *,
        matched_problem: ExactProblemMatch | None = None,
    ) -> tuple[JsonMap, ...]:
        text = understanding.normalized_query.lower()
        linked: list[JsonMap] = []
        if matched_problem is not None:
            self._append_linked(
                linked,
                {
                    "entityId": matched_problem.problem_id,
                    "name": matched_problem.title,
                    "type": "problem",
                    "confidence": matched_problem.confidence,
                    "matchKind": matched_problem.match_kind,
                },
            )
        if understanding.code_features is not None:
            for feature in understanding.code_features.features:
                entity = _CODE_FEATURE_ENTITY_MAP.get(feature)
                if entity is None:
                    continue
                self._append_linked(
                    linked,
                    {
                        **entity,
                        "confidence": 1.0,
                        "matchedBy": "code_feature",
                        "codeFeatureNodeId": f"code_feature:{feature}",
                    },
                )
        for entity_id in understanding.query_variants.get("graphSeeds", ()):
            metadata = GRAPH_SEED_ENTITY_REGISTRY.get(str(entity_id))
            if metadata is None:
                continue
            self._append_linked(
                linked,
                {
                    **metadata,
                    "confidence": 1.0,
                    "matchedBy": "concept_seed",
                },
            )
        for entity_id, name, kind in self._concept_aliases:
            terms = _tokens(name)
            if any(term in understanding.keywords for term in terms) or name.lower() in text:
                self._append_linked(
                    linked,
                    {
                        "entityId": entity_id,
                        "name": name,
                        "type": kind,
                        "confidence": 1.0,
                    },
                )
        return tuple(linked)

    @staticmethod
    def _append_linked(linked: list[JsonMap], entity: JsonMap) -> None:
        for existing in linked:
            if existing.get("entityId") != entity.get("entityId"):
                continue
            existing["confidence"] = max(
                float(existing.get("confidence", 0.0)),
                float(entity.get("confidence", 0.0)),
            )
            for key in ("name", "type", "matchKind"):
                if key in entity and not existing.get(key):
                    existing[key] = entity[key]
            if entity.get("matchedBy") == "code_feature":
                existing["matchedBy"] = "code_feature"
                existing["codeFeatureNodeId"] = entity["codeFeatureNodeId"]
            return
        linked.append(dict(entity))


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
        vector_query = str(understanding.query_variants.get("vector") or understanding.normalized_query)
        query_vector = self._embedding_provider.embed_text(vector_query)
        if self._vector_store is not None:
            return _search_and_aggregate_store_candidates(
                lambda window: self._vector_store.search(query_vector, top_k=window),
                source="vector",
                top_k=top_k,
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
        bm25_query = str(understanding.query_variants.get("bm25") or understanding.normalized_query)
        if self._bm25_store is not None:
            return _search_and_aggregate_store_candidates(
                lambda window: self._bm25_store.search(
                    bm25_query,
                    top_k=window,
                ),
                source="bm25",
                top_k=top_k,
            )

        query_terms = set(shared_multilingual_tokens(bm25_query))
        candidates: list[RetrievalCandidate] = []
        for document in self._documents:
            terms = _tokens(
                " ".join(
                    (
                        document.id,
                        document.source,
                        document.source_id,
                        document.title,
                        " ".join(_problem_aliases(document)),
                        document.text,
                        document.answer,
                        " ".join(document.concepts),
                    )
                )
            )
            if not query_terms:
                score = 0.0
            else:
                score = sum(1 for term in terms if term in query_terms) / max(len(terms), 1)
                score += len(query_terms & set(terms)) / len(query_terms)
            if score > 0:
                candidates.append(_candidate_from_document(document, source="bm25", score=score))
        return tuple(sorted(candidates, key=lambda item: (-item.score, item.id))[:top_k])


class GraphSearchService:
    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        graph_store: GraphQueryStore | None = None,
    ) -> None:
        self._documents = tuple(documents)
        self._graph_store = graph_store

    def search(
        self,
        linked_entities: Sequence[JsonMap],
        *,
        top_k: int,
        matched_problem: ExactProblemMatch | None = None,
    ) -> GraphSearchResult:
        if self._graph_store is not None:
            return self._search_store(
                linked_entities,
                top_k=top_k,
                matched_problem=matched_problem,
            )

        if matched_problem is not None:
            document = self._document_for_match(matched_problem)
            if document is not None:
                return GraphSearchResult(
                    candidates=(),
                    paths=_inferred_problem_paths(document),
                )

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
                    _canonical_graph_path(
                        document,
                        target_node=str(match["entityId"]),
                        target_label=str(match["name"]),
                        path_source="inferred",
                        operation="candidate_retrieval",
                        edge_weight=score,
                        rationale=f"linked {match['name']} to {document.title}",
                    )
                )
        return GraphSearchResult(
            candidates=tuple(sorted(candidates, key=lambda item: (-item.score, item.id))[:top_k]),
            paths=tuple(paths),
        )

    def _search_store(
        self,
        linked_entities: Sequence[JsonMap],
        *,
        top_k: int,
        matched_problem: ExactProblemMatch | None = None,
    ) -> GraphSearchResult:
        assert self._graph_store is not None
        if matched_problem is not None:
            document = self._document_for_match(matched_problem)
            if document is not None:
                paths: list[JsonMap] = []
                target_nodes = [
                    (f"concept:{_slug(concept)}", concept)
                    for concept in document.concepts
                ]
                if document.problem_type:
                    target_nodes.append(
                        (
                            f"pattern:{_slug(document.problem_type)}",
                            document.problem_type,
                        )
                    )
                inferred_paths_by_target = {
                    _graph_path_target_id(path): path
                    for path in _inferred_problem_paths(document)
                }
                for target_node, target_name in target_nodes:
                    direct_paths = self._graph_store.find_paths(
                        document.id,
                        target_node,
                        max_hops=3,
                    )
                    reverse_paths = self._graph_store.find_paths(
                        target_node,
                        document.id,
                        max_hops=3,
                    )
                    best_normalized: JsonMap | None = None
                    for path in (*direct_paths, *reverse_paths):
                        normalized = _normalize_graph_store_path(
                            path,
                            entity={
                                "entityId": target_node,
                                "name": target_name,
                            },
                            document=document,
                            source_node=document.id,
                            target_node=target_node,
                        )
                        if best_normalized is None or (
                            not _has_usable_store_path(best_normalized)
                            and _has_usable_store_path(normalized)
                        ):
                            best_normalized = normalized
                    if best_normalized is not None:
                        paths.append(best_normalized)
                    else:
                        inferred_path = inferred_paths_by_target.get(target_node)
                        if inferred_path is not None:
                            paths.append(inferred_path)
                return GraphSearchResult(
                    candidates=(),
                    paths=tuple(paths),
                )

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

                if entity_id in document_paths_by_entity:
                    continue
                related_problem_ids = (
                    self._graph_store.find_related_problem_ids(
                        entity_id,
                        top_k=top_k,
                    )
                    if isinstance(self._graph_store, GraphRelatedProblemLookup)
                    else ()
                )
                if document.id not in related_problem_ids:
                    continue
                entity_name = str(entity.get("name", entity_id))
                metadata_path = _canonical_graph_path(
                    document,
                    target_node=entity_id,
                    target_label=entity_name,
                    path_source="neo4j",
                    operation="candidate_retrieval",
                    edge_weight=1.0,
                    rationale=(
                        f"Neo4j metadata linked concept {entity_name} "
                        f"to {document.title}."
                    ),
                )
                metadata_path["storePath"] = {
                    "nodes": [entity_id, document.id],
                    "relations": ["RELATED_PROBLEM_ID"],
                }
                document_paths_by_entity.setdefault(entity_id, []).append(metadata_path)
                paths.append(metadata_path)

            if not document_paths_by_entity:
                continue
            score = min(len(document_paths_by_entity) / linked_entity_count, 1.0)
            candidates.append(_candidate_from_document(document, source="graph", score=score))

        ranked_candidates = sorted(candidates, key=lambda item: (-item.score, item.id))
        candidate_rank = {
            candidate.id: index
            for index, candidate in enumerate(ranked_candidates)
        }
        paths.sort(
            key=lambda path: candidate_rank.get(
                _graph_path_problem_id(path),
                len(candidate_rank),
            )
        )
        return GraphSearchResult(
            candidates=tuple(ranked_candidates[:top_k]),
            paths=tuple(paths),
        )

    def _document_for_match(
        self,
        matched_problem: ExactProblemMatch,
    ) -> RetrievalDocument | None:
        return next(
            (
                document
                for document in self._documents
                if document.id == matched_problem.problem_id
            ),
            None,
        )


def _chunk_identity(chunk: JsonMap) -> str:
    payload = _mapping(chunk.get("payload"))
    store_candidate_id = payload.get("storeCandidateId") or payload.get("store_candidate_id")
    if store_candidate_id:
        return f"store:{store_candidate_id}"

    chunk_id = chunk.get("id")
    title = chunk.get("title")
    if chunk_id or title:
        return f"chunk:{chunk_id or ''}:{title or ''}"

    return f"serialized:{json.dumps(chunk, sort_keys=True, default=str)}"


def _raw_chunks_for_candidate(candidate: RetrievalCandidate) -> list[JsonMap]:
    if "rawChunks" in candidate.payload:
        raw_chunks = candidate.payload["rawChunks"]
    elif "raw_chunks" in candidate.payload:
        raw_chunks = candidate.payload["raw_chunks"]
    else:
        raw_chunks = None

    if isinstance(raw_chunks, (list, tuple)) and raw_chunks:
        return [deepcopy(chunk) for chunk in raw_chunks if isinstance(chunk, dict)]

    return [deepcopy(candidate.to_mapping())]


def _raw_chunk_payload(chunk: JsonMap) -> JsonMap:
    payload = _mapping(chunk.get("payload"))
    store_payload = payload.get("storePayload") or payload.get("store_payload")
    if isinstance(store_payload, dict):
        return dict(store_payload)
    if payload:
        return payload
    return _mapping(chunk)


def _chunk_kind(chunk: JsonMap) -> str:
    payload = _raw_chunk_payload(chunk)
    return str(payload.get("kind") or chunk.get("kind") or "").strip().lower()


def _chunk_display_text(chunk: JsonMap) -> str:
    payload = _raw_chunk_payload(chunk)
    for key in ("displayText", "display_text", "text"):
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _chunk_score(chunk: JsonMap) -> float:
    payload = _raw_chunk_payload(chunk)
    value = chunk.get("score", payload.get("score", 0.0))
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _chunk_selection_mapping(chunk: JsonMap) -> JsonMap:
    payload = _raw_chunk_payload(chunk)
    return {
        "id": str(
            chunk.get("id")
            or payload.get("storeCandidateId")
            or payload.get("store_candidate_id")
            or ""
        ),
        "kind": _chunk_kind(chunk),
        "displayText": _chunk_display_text(chunk),
        "score": _chunk_score(chunk),
    }


def _candidate_chunk_by_kind(
    candidate: RetrievalCandidate,
    kinds: set[str],
) -> JsonMap | None:
    eligible = [
        chunk
        for chunk in _raw_chunks_for_candidate(candidate)
        if _chunk_kind(chunk) in kinds and _chunk_display_text(chunk)
    ]
    if not eligible:
        return None
    selected = sorted(
        eligible,
        key=lambda chunk: (-_chunk_score(chunk), str(chunk.get("id") or "")),
    )[0]
    return _chunk_selection_mapping(selected)


def _candidate_problem_card(candidate: RetrievalCandidate) -> JsonMap | None:
    return _candidate_chunk_by_kind(candidate, {"problem_card"})


def _candidate_statement(candidate: RetrievalCandidate) -> JsonMap | None:
    return _candidate_chunk_by_kind(candidate, {"statement"})


def _candidate_solution(candidate: RetrievalCandidate) -> JsonMap | None:
    return _candidate_chunk_by_kind(candidate, {"solution", "answer", "editorial"})


def _best_matched_chunk(candidate: RetrievalCandidate) -> JsonMap | None:
    eligible = [
        chunk
        for chunk in _raw_chunks_for_candidate(candidate)
        if _chunk_kind(chunk) not in {"problem_card", "common_mistakes"}
        and _chunk_display_text(chunk)
    ]
    if not eligible:
        return None
    selected = sorted(
        eligible,
        key=lambda chunk: (-_chunk_score(chunk), str(chunk.get("id") or "")),
    )[0]
    return _chunk_selection_mapping(selected)


def _append_common_mistakes(
    mistakes: list[str],
    value: Any,
    *,
    split_lines: bool = False,
) -> None:
    values = str(value).splitlines() if split_lines and isinstance(value, str) else _tuple_of_str(value)
    for item in values:
        mistake = str(item).strip()
        if split_lines:
            mistake = mistake.lstrip("-").strip()
        if mistake:
            _append_unique(mistakes, mistake)


def _append_common_mistakes_from_payload(mistakes: list[str], payload: JsonMap) -> None:
    _append_common_mistakes(
        mistakes,
        payload.get("commonMistakes") or payload.get("common_mistakes"),
    )
    metadata = _mapping(payload.get("metadata"))
    _append_common_mistakes(
        mistakes,
        metadata.get("commonMistakes") or metadata.get("common_mistakes"),
    )


def _common_mistakes_from_candidate(candidate: RetrievalCandidate) -> list[str]:
    mistakes: list[str] = []
    candidate_payload = _mapping(candidate.payload)
    _append_common_mistakes_from_payload(mistakes, candidate_payload)
    store_payload = candidate_payload.get("storePayload") or candidate_payload.get("store_payload")
    if isinstance(store_payload, dict):
        _append_common_mistakes_from_payload(mistakes, store_payload)

    for chunk in _raw_chunks_for_candidate(candidate):
        chunk_payload = _raw_chunk_payload(chunk)
        _append_common_mistakes_from_payload(mistakes, chunk_payload)
        if _chunk_kind(chunk) == "common_mistakes":
            _append_common_mistakes(
                mistakes,
                _chunk_display_text(chunk),
                split_lines=True,
            )
    return mistakes


def _fallback_common_mistakes() -> list[str]:
    return list(_FALLBACK_COMMON_MISTAKES)


def _score_meta(stage: str) -> JsonMap:
    labels = {
        "vector": "Vector similarity",
        "graph": "Graph match",
        "bm25": "BM25 lexical score",
        "fusion": "Hybrid fusion score",
        "reranker": "Reranker score",
        "graph_path": "Graph path confidence",
    }
    return ScoreMetadata(
        stage=stage,
        display_label=labels.get(stage, "Retrieval score"),
    ).to_mapping()


def _candidate_mapping(candidate: RetrievalCandidate, *, stage: str) -> JsonMap:
    mapping = candidate.to_mapping()
    mapping["scoreMeta"] = _score_meta(stage)
    return mapping


def _chunk_evidence(
    sources: Sequence[str],
    raw_chunks: Sequence[JsonMap],
    complete_by_source: JsonMap,
) -> JsonMap:
    chunk_sources = {
        str(chunk.get("source") or chunk.get("candidateSource") or "")
        for chunk in raw_chunks
    }
    missing_sources = [
        source
        for source in sources
        if source in {"vector", "bm25"} and source not in chunk_sources
    ]
    available = len(raw_chunks) > 0
    complete = available and not missing_sources and all(
        bool(complete_by_source.get(source, True))
        for source in sources
    )
    unavailable_reason = ""
    if not available:
        unavailable_reason = "raw chunks unavailable"
    elif missing_sources:
        unavailable_reason = "raw chunks incomplete"

    return {
        "available": available,
        "complete": complete,
        "missingSources": missing_sources,
        "unavailableReason": unavailable_reason,
    }


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
        raw_chunks_by_id: dict[str, list[JsonMap]] = {}
        raw_chunk_identities: dict[str, set[str]] = {}
        complete_by_source: dict[str, dict[str, bool]] = {}
        for source, candidates in groups.items():
            best_by_id: dict[str, RetrievalCandidate] = {}
            for candidate in candidates:
                if candidate.score <= 0:
                    continue
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
                complete_by_source.setdefault(candidate.id, {})[source] = bool(
                    candidate.payload.get("rawChunksComplete", True)
                )
                raw_chunks = raw_chunks_by_id.setdefault(candidate.id, [])
                seen = raw_chunk_identities.setdefault(candidate.id, set())
                for chunk in _raw_chunks_for_candidate(candidate):
                    chunk.setdefault("source", source)
                    identity = _chunk_identity(chunk)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    raw_chunks.append(chunk)

        fused: list[RetrievalCandidate] = []
        for candidate_id, candidate in by_id.items():
            ordered_sources = sorted(sources[candidate_id], key=self._source_order.get)
            payload = dict(candidate.payload)
            raw_chunks = raw_chunks_by_id.get(candidate_id, [])
            payload["sources"] = ordered_sources
            payload["rawChunks"] = deepcopy(raw_chunks)
            payload["chunkCount"] = len(raw_chunks)
            chunk_evidence = _chunk_evidence(
                ordered_sources,
                raw_chunks,
                complete_by_source.get(candidate_id, {}),
            )
            payload["rawChunksComplete"] = chunk_evidence["complete"]
            payload["chunkEvidence"] = chunk_evidence
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
        query_understanding: QueryUnderstanding | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        query_terms = set(_tokens(query))
        reranked: list[RetrievalCandidate] = []
        for candidate in candidates:
            candidate_terms = set(
                _tokens(f"{candidate.title} {candidate.text} {' '.join(candidate.concepts)}")
            )
            lexical = len(query_terms & candidate_terms) / max(len(query_terms), 1)
            compatibility = _code_feature_compatibility(query_understanding, candidate)
            if query_understanding is not None and query_understanding.code_features is not None:
                reranker_score = round(
                    min(
                        (0.55 * lexical)
                        + (0.25 * candidate.score)
                        + (0.20 * compatibility),
                        1.0,
                    ),
                    6,
                )
            else:
                reranker_score = round((0.7 * lexical) + (0.3 * candidate.score), 6)
            payload = dict(candidate.payload)
            payload["rerankerScore"] = reranker_score
            if query_understanding is not None and query_understanding.code_features is not None:
                payload["codeFeatureCompatibility"] = compatibility
            reranked.append(replace(candidate, score=reranker_score, payload=payload))
        return tuple(sorted(reranked, key=lambda item: (-item.score, item.id))[:top_k])


class EvidenceBuilder:
    def build(
        self,
        candidates: Sequence[RetrievalCandidate],
        graph_paths: Sequence[JsonMap],
        *,
        matched_problem: ExactProblemMatch | None = None,
    ) -> RetrievalEvidenceBundle:
        matched_problem_id = matched_problem.problem_id if matched_problem else ""
        similar_candidates = [candidate for candidate in candidates if candidate.id != matched_problem_id]
        algorithms: list[str] = []
        data_structures: list[str] = []
        patterns: list[str] = []
        techniques: list[str] = []
        evidence_candidates = list(candidates)
        if matched_problem and all(candidate.id != matched_problem.problem_id for candidate in evidence_candidates):
            evidence_candidates.append(matched_problem.candidate)
        for candidate in evidence_candidates:
            for concept in candidate.concepts:
                kind = _classify_concept(concept)
                if kind == "algorithm":
                    _append_unique(algorithms, concept)
                elif kind == "data_structure":
                    _append_unique(data_structures, concept)
                elif kind == "technique":
                    _append_unique(techniques, concept)
            if candidate.problem_type:
                _append_unique(patterns, candidate.problem_type)

        common_mistakes: list[str] = []
        for candidate in evidence_candidates:
            for mistake in _common_mistakes_from_candidate(candidate):
                _append_unique(common_mistakes, mistake)
        if not common_mistakes:
            common_mistakes = _fallback_common_mistakes()

        matched_problem_mapping = matched_problem.to_mapping() if matched_problem else None
        if matched_problem_mapping is not None:
            matched_problem_mapping.update(
                {
                    "problemCard": _candidate_problem_card(matched_problem.candidate),
                    "statement": _candidate_statement(matched_problem.candidate),
                    "solution": _candidate_solution(matched_problem.candidate),
                }
            )

        return RetrievalEvidenceBundle(
            similar_problems=[
                {
                    "id": candidate.id,
                    "title": candidate.title,
                    "score": round(candidate.score, 6),
                    "sharedConcepts": list(candidate.concepts),
                    "answerHint": candidate.payload.get("answer", ""),
                    "solutionHints": list(_tuple_of_str(candidate.payload.get("solutionHints"))),
                    "difficulty": str(candidate.payload.get("difficulty") or ""),
                    "constraints": list(_tuple_of_str(candidate.payload.get("constraints"))),
                    "source": str(candidate.payload.get("documentSource") or ""),
                    "sourceId": str(candidate.payload.get("sourceId") or ""),
                    "problemCard": _candidate_problem_card(candidate),
                    "matchedChunk": _best_matched_chunk(candidate),
                }
                for candidate in similar_candidates
            ],
            graph_paths=[dict(path) for path in graph_paths],
            algorithm_evidence=algorithms,
            data_structure_evidence=data_structures,
            pattern_evidence=patterns,
            technique_evidence=techniques,
            matched_problem=matched_problem_mapping,
            common_mistakes=common_mistakes,
        )


class ContextBuilder:
    def build(
        self,
        query_understanding: QueryUnderstanding,
        evidence_bundle: RetrievalEvidenceBundle,
    ) -> str:
        evidence = evidence_bundle.to_mapping()
        lines = [
            "查詢理解",
            f"- 意圖: {query_understanding.intent}",
            f"- 輸入類型: {query_understanding.input_kind}",
            f"- 關鍵詞: {', '.join(query_understanding.keywords)}",
            "",
            "命中題目",
        ]
        matched_problem = evidence.get("matchedProblem")
        if matched_problem:
            lines.extend(
                [
                    f"- id: {matched_problem['id']}",
                    f"  title: {matched_problem['title']}",
                    f"  matchKind: {matched_problem['matchKind']}",
                    f"  confidence: {matched_problem['confidence']}",
                ]
            )
            if matched_problem.get("answerHint"):
                lines.append(f"  答案摘要: {matched_problem['answerHint']}")
            for hint in matched_problem.get("solutionHints", []):
                lines.append(f"  解題提示: {hint}")
            for key, label in (
                ("problemCard", "problemCard"),
                ("statement", "statement"),
                ("solution", "solution"),
            ):
                selected = matched_problem.get(key)
                if isinstance(selected, dict) and selected.get("displayText"):
                    lines.append(f"  {label}: {selected['displayText']}")
        else:
            lines.append("- 無")
        if evidence["similarProblems"]:
            lines.extend(["", "相似題"])
            for problem in evidence["similarProblems"]:
                lines.append(
                    f"- {problem['id']} {problem['title']} "
                    f"(score={problem['score']}, concepts={', '.join(problem['sharedConcepts'])})"
                )
                if problem.get("answerHint"):
                    lines.append(f"  答案摘要: {problem['answerHint']}")
                for hint in problem.get("solutionHints", []):
                    lines.append(f"  解題提示: {hint}")
                if problem.get("difficulty"):
                    lines.append(f"  難度: {problem['difficulty']}")
                if problem.get("constraints"):
                    lines.append(f"  限制: {', '.join(problem['constraints'])}")
                problem_card = problem.get("problemCard")
                if isinstance(problem_card, dict) and problem_card.get("displayText"):
                    lines.append(f"  problemCard: {problem_card['displayText']}")
                matched_chunk = problem.get("matchedChunk")
                if isinstance(matched_chunk, dict) and matched_chunk.get("displayText"):
                    lines.append(f"  matchedChunk: {matched_chunk['displayText']}")
        lines.extend(["", "圖路徑"])
        for path in evidence["graphPaths"]:
            nodes = [_graph_path_node_label(node) for node in path.get("nodes", [])]
            relations = [
                _graph_path_relation_label(relation)
                for relation in path.get("relations", [])
            ]
            rationale = str(path.get("rationale", ""))
            lines.append(
                f"- {' -> '.join(nodes)} "
                f"(relations={', '.join(relations)}, rationale={rationale})"
            )
        lines.extend(
            [
                "",
                "演算法證據",
                f"- {', '.join(evidence['algorithmEvidence'])}",
                "資料結構證據",
                f"- {', '.join(evidence['dataStructureEvidence'])}",
                "技巧證據",
                f"- {', '.join(evidence.get('techniqueEvidence', []))}",
                "題型證據",
                f"- {', '.join(evidence['patternEvidence'])}",
                "常見錯誤",
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
        graph_store: GraphQueryStore | None = None,
    ) -> None:
        self._documents = tuple(documents) if documents is not None else _load_default_documents()
        self._embedding_provider = embedding_provider or DeterministicMockEmbeddingProvider()
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._graph_store = graph_store

    def run(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid",
        top_k: int = 5,
    ) -> OnlineQueryResult:
        understanding = QueryUnderstandingService(self._documents).understand(query)
        matched_problem = ExactProblemMatcher(self._documents).match(understanding)
        if (
            matched_problem
            and matched_problem.match_kind == "partial_title"
            and understanding.input_kind in {"cpp", "python"}
        ):
            matched_problem = None
        linked_entities = EntityLinkingService().link(understanding, matched_problem=matched_problem)
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
            matched_problem=matched_problem,
        )
        fusion_inputs = {
            "vector_candidates": vector_candidates if mode in {"hybrid", "vector"} else (),
            "graph_candidates": graph_result.candidates if mode in {"hybrid", "graph"} else (),
            "bm25_candidates": bm25_candidates if mode == "hybrid" else (),
        }
        fused = HybridFusionService().fuse(
            **fusion_inputs,
            top_k=max(top_k * 2, top_k),
        )
        reranked = Reranker().rerank(
            understanding.normalized_query,
            fused,
            top_k=top_k,
            query_understanding=understanding,
        )
        target_problem_ids = (
            (
                matched_problem.problem_id,
                matched_problem.source_id,
            )
            if matched_problem is not None
            else ()
        ) + tuple(candidate.id for candidate in reranked)
        pruned_graph_paths = _prune_graph_paths(
            graph_result.paths,
            target_problem_ids,
        )
        graph_search_status = _graph_search_status(
            graph_result.candidates,
            pruned_graph_paths or graph_result.paths,
        )
        trace = RetrievalTrace(
            query_understanding=understanding.to_mapping(),
            entity_linking=[dict(entity) for entity in linked_entities],
            vector_candidates=[
                _candidate_mapping(candidate, stage="vector")
                for candidate in vector_candidates
            ],
            graph_candidates=[
                _candidate_mapping(candidate, stage="graph")
                for candidate in graph_result.candidates
            ],
            graph_search_status=graph_search_status,
            bm25_candidates=[
                _candidate_mapping(candidate, stage="bm25")
                for candidate in bm25_candidates
            ],
            fusion_scores=[
                _candidate_mapping(candidate, stage="fusion")
                for candidate in fused
            ],
            reranker_scores=[
                _candidate_mapping(candidate, stage="reranker")
                for candidate in reranked
            ],
            matched_problem=matched_problem.to_mapping() if matched_problem is not None else None,
        )
        return OnlineQueryResult(
            query_understanding=understanding,
            linked_entities=linked_entities,
            matched_problem=matched_problem,
            vector_candidates=vector_candidates,
            graph_candidates=graph_result.candidates,
            bm25_candidates=bm25_candidates,
            fused_candidates=fused,
            reranked_candidates=reranked,
            graph_paths=pruned_graph_paths,
            trace=trace,
            raw_graph_paths=graph_result.paths,
        )


def _candidate_from_document(
    document: RetrievalDocument,
    *,
    source: str,
    score: float,
) -> RetrievalCandidate:
    payload = _candidate_evidence_payload(
        base_payload={},
        metadata={},
        fallback=document,
    )
    if source == "exact":
        raw_chunks = _exact_document_chunks(document)
        payload.update(
            {
                "sources": ["exact"],
                "rawChunks": raw_chunks,
                "chunkCount": len(raw_chunks),
                "rawChunksComplete": True,
                "chunkEvidence": _chunk_evidence(
                    ("exact",),
                    raw_chunks,
                    {"exact": True},
                ),
            }
        )

    return RetrievalCandidate(
        id=document.id,
        title=document.title,
        source=source,
        score=round(score, 6),
        text=document.text,
        concepts=document.concepts,
        problem_type=document.problem_type,
        payload=payload,
    )


def _exact_document_chunks(document: RetrievalDocument) -> list[JsonMap]:
    chunks: list[JsonMap] = []

    def append_chunk(kind: str, store_candidate_id: str, text: str) -> None:
        if not text:
            return
        chunks.append(
            {
                "id": store_candidate_id,
                "title": document.title,
                "source": "exact",
                "score": 1.0,
                "concepts": list(document.concepts),
                "problemType": document.problem_type,
                "payload": {
                    "storeCandidateId": store_candidate_id,
                    "kind": kind,
                    "text": text,
                    "documentSource": document.source,
                    "sourceId": document.source_id,
                    "title": document.title,
                    "problemType": document.problem_type,
                    "concepts": list(document.concepts),
                },
            }
        )

    problem_card = " - ".join(
        part
        for part in (document.source, document.source_id, document.title)
        if part
    )
    append_chunk("problem_card", f"{document.id}:problem_card:0", problem_card)
    append_chunk("statement", f"{document.id}:statement:0", document.text)
    append_chunk("answer", f"{document.id}:answer:0", document.answer)
    for index, hint in enumerate(document.solution_hints, start=1):
        append_chunk("hint", f"{document.id}:hint:{index}", hint)

    return chunks


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
    evidence_payload = _candidate_evidence_payload(
        base_payload=payload,
        metadata=metadata,
    )
    concepts = tuple(str(item) for item in evidence_payload["concepts"])
    title = str(evidence_payload["title"] or problem_id)
    problem_type = str(evidence_payload["problemType"])
    text = str(payload.get("text") or payload.get("statement") or "")

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
            **evidence_payload,
            "storePayload": payload,
        },
    )


def _aggregate_problem_candidates(
    candidates: Sequence[RetrievalCandidate],
    *,
    source: str,
    top_k: int,
    raw_chunks_complete: bool = False,
) -> tuple[RetrievalCandidate, ...]:
    grouped: dict[str, list[RetrievalCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.id, []).append(candidate)

    aggregated: list[RetrievalCandidate] = []
    for problem_candidates in grouped.values():
        ordered_chunks = sorted(
            problem_candidates,
            key=lambda item: (-item.score, str(item.payload.get("storeCandidateId", ""))),
        )
        best = ordered_chunks[0]
        payload = deepcopy(best.payload)
        payload["rawChunks"] = [deepcopy(chunk.to_mapping()) for chunk in ordered_chunks]
        payload["chunkCount"] = len(ordered_chunks)
        payload["rawChunksComplete"] = raw_chunks_complete
        payload["sources"] = [source]
        aggregated.append(replace(best, score=round(best.score, 6), payload=payload))

    return tuple(sorted(aggregated, key=lambda item: (-item.score, item.id))[:top_k])


def _search_and_aggregate_store_candidates(
    search: Callable[[int], Sequence[SearchCandidate]],
    *,
    source: str,
    top_k: int,
) -> tuple[RetrievalCandidate, ...]:
    if top_k <= 0:
        return ()

    window = min(top_k, _MAX_STORE_FETCH_WINDOW)
    raw_by_id: dict[str, SearchCandidate] = {}
    aggregated: tuple[RetrievalCandidate, ...] = ()
    enrichment_requested = False
    for _ in range(_MAX_STORE_FETCH_ATTEMPTS):
        rows = tuple(search(window))
        previous_count = len(raw_by_id)
        for row in rows:
            raw_by_id.setdefault(row.id, row)

        raw_candidates = tuple(
            _candidate_from_store_candidate(candidate, source=source)
            for candidate in raw_by_id.values()
            if candidate.score > 0
        )
        aggregated = _aggregate_problem_candidates(
            raw_candidates,
            source=source,
            top_k=top_k,
            raw_chunks_complete=len(rows) < window,
        )
        if len(rows) < window or len(raw_by_id) == previous_count:
            break

        next_window = min(window * 2, _MAX_STORE_FETCH_WINDOW)
        if len(aggregated) >= top_k:
            if enrichment_requested or next_window == window:
                break
            enrichment_requested = True
        elif next_window == window:
            break
        window = next_window

    return aggregated


def _candidate_evidence_payload(
    *,
    base_payload: JsonMap,
    metadata: JsonMap,
    fallback: RetrievalDocument | None = None,
) -> JsonMap:
    title = base_payload.get("title") or metadata.get("title") or (fallback.title if fallback else "")
    problem_type = (
        base_payload.get("problemType")
        or base_payload.get("problem_type")
        or metadata.get("problemType")
        or (fallback.problem_type if fallback else "")
    )
    concepts = _tuple_of_str(
        base_payload.get("concepts")
        or metadata.get("concepts")
        or (fallback.concepts if fallback else ())
    )
    answer = base_payload.get("answer") or metadata.get("answer") or (fallback.answer if fallback else "")
    source = base_payload.get("source") or metadata.get("source") or (fallback.source if fallback else "")
    source_id = (
        base_payload.get("sourceId")
        or metadata.get("sourceId")
        or (fallback.source_id if fallback else "")
    )
    difficulty = (
        base_payload.get("difficulty")
        or metadata.get("difficulty")
        or (fallback.difficulty if fallback else "")
    )
    editorial = (
        base_payload.get("editorial")
        or metadata.get("editorial")
        or (fallback.editorial if fallback else "")
    )
    solution_hints = _tuple_of_str(
        base_payload.get("solutionHints")
        or base_payload.get("solution_hints")
        or metadata.get("solutionHints")
        or (fallback.solution_hints if fallback else ())
    )
    constraints = _tuple_of_str(
        base_payload.get("constraints")
        or metadata.get("constraints")
        or (fallback.constraints if fallback else ())
    )
    examples = _tuple_of_mapping(
        base_payload.get("examples")
        or metadata.get("examples")
        or (fallback.examples if fallback else ())
    )
    return {
        "documentSource": str(source),
        "sourceId": str(source_id),
        "answer": str(answer),
        "solutionHints": list(solution_hints),
        "difficulty": str(difficulty),
        "constraints": list(constraints),
        "examples": [dict(example) for example in examples],
        "editorial": str(editorial),
        "title": str(title),
        "problemType": str(problem_type),
        "concepts": list(concepts),
    }


def _normalize_graph_store_path(
    path: JsonMap,
    *,
    entity: JsonMap,
    document: RetrievalDocument,
    source_node: str | None = None,
    target_node: str | None = None,
) -> JsonMap:
    entity_id = str(entity["entityId"])
    entity_name = str(entity.get("name", entity_id))
    raw_nodes = _list_or_tuple_of_str(path.get("nodes"))
    raw_relations = _list_or_tuple_of_str(path.get("relations"))
    raw_score = _clamp_graph_weight(path.get("score", 0.0))
    is_exact_problem_path = source_node is not None and target_node is not None
    if is_exact_problem_path:
        public_target_node = target_node
        operation = "exact_expansion"
        rationale = (
            f"Neo4j returned a path from {document.title} "
            f"to {entity_name}."
        )
    else:
        public_target_node = entity_id
        operation = "candidate_retrieval"
        rationale = f"Neo4j linked {entity_name} to {document.title}."

    normalized = _canonical_graph_path(
        document,
        target_node=public_target_node,
        target_label=entity_name,
        path_source="neo4j",
        operation=operation,
        edge_weight=raw_score,
        raw_relation=raw_relations[-1] if raw_relations else None,
        rationale=rationale,
    )
    normalized["storePath"] = {
        "nodes": raw_nodes,
        "relations": raw_relations,
    }
    return normalized


def _has_usable_store_path(path: JsonMap) -> bool:
    store_path = path.get("storePath")
    if not isinstance(store_path, dict):
        return False
    return bool(store_path.get("nodes")) and bool(store_path.get("relations"))


def _clamp_graph_weight(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return round(min(max(score, 0.0), 1.0), 6)


def _list_or_tuple_of_str(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value]


def _inferred_problem_paths(document: RetrievalDocument) -> tuple[JsonMap, ...]:
    paths = [
        _canonical_graph_path(
            document,
            target_node=f"concept:{_slug(concept)}",
            target_label=concept,
            path_source="inferred",
            operation="exact_expansion",
            edge_weight=1.0,
            rationale=(
                f"Inferred from document concepts; this path was not returned by Neo4j "
                f"for {document.title} and {concept}."
            ),
        )
        for concept in document.concepts
    ]
    if document.problem_type:
        paths.append(
            _canonical_graph_path(
                document,
                target_node=f"pattern:{_slug(document.problem_type)}",
                target_label=document.problem_type,
                path_source="inferred",
                operation="exact_expansion",
                edge_weight=1.0,
                rationale=(
                    "Inferred from document problem_type; this path was not returned "
                    f"by Neo4j for {document.title} and {document.problem_type}."
                ),
            )
        )
    return tuple(paths)


def _canonical_graph_path(
    document: RetrievalDocument,
    *,
    target_node: str,
    target_label: str,
    path_source: Literal["inferred", "neo4j"],
    operation: Literal["candidate_retrieval", "exact_expansion"],
    edge_weight: Any,
    rationale: str,
    raw_relation: str | None = None,
) -> JsonMap:
    problem_node = _graph_node(document.id, label=document.title, layer="problem")
    source_node = _graph_node(
        _source_node_id(document),
        label=_source_node_label(document),
        layer="source",
    )
    target = _graph_node(
        target_node,
        label=target_label,
        layer=_graph_layer_for_node(target_node),
    )
    source_relation_type = (
        "EXPANDED_FROM_EXACT_MATCH"
        if operation == "exact_expansion"
        else "DERIVED_FROM_SOURCE"
    )
    target_relation_type = _canonical_target_relation_type(
        target,
        target_label,
        raw_relation,
    )
    target_relation = _graph_edge(source_node, target, target_relation_type, edge_weight)
    raw_relation_type = str(raw_relation or "").upper()
    if raw_relation_type and raw_relation_type not in _ALLOWED_GRAPH_RELATION_TYPES:
        target_relation["normalizedFrom"] = raw_relation_type

    relations = [
        _graph_edge(problem_node, source_node, source_relation_type, 1.0),
        target_relation,
    ]
    nodes = [problem_node, source_node, target]
    scoring = _score_graph_path(
        nodes,
        relations,
        path_source=path_source,
    )
    return {
        "nodes": nodes,
        "relations": relations,
        "score": scoring["score"],
        "rationale": rationale,
        "pathSource": path_source,
        "graphPathOperation": operation,
        "pathScoring": scoring,
        "scoreMeta": _score_meta("graph_path"),
    }


def _graph_node(node_id: str, *, label: str, layer: str) -> JsonMap:
    canonical_layer = layer if layer in _ALLOWED_GRAPH_NODE_LAYERS else "concept"
    return {
        "id": str(node_id),
        "label": str(label or node_id),
        "layer": canonical_layer,
    }


def _graph_edge(source: JsonMap, target: JsonMap, edge_type: str, weight: Any) -> JsonMap:
    raw_type = str(edge_type or "").upper()
    canonical_type = (
        raw_type if raw_type in _ALLOWED_GRAPH_RELATION_TYPES else "SIMILAR_BY_FEATURE"
    )
    edge = {
        "source": str(source["id"]),
        "target": str(target["id"]),
        "type": canonical_type,
        "weight": _clamp_graph_weight(weight),
    }
    if raw_type and canonical_type != raw_type:
        edge["normalizedFrom"] = raw_type
    return edge


def _score_graph_path(
    nodes: Sequence[JsonMap],
    relations: Sequence[JsonMap],
    *,
    path_source: Literal["inferred", "neo4j"],
) -> JsonMap:
    weights = [_clamp_graph_weight(relation.get("weight", 0.0)) for relation in relations]
    min_edge_weight = min(weights) if weights else 0.0
    mean_edge_weight = sum(weights) / len(weights) if weights else 0.0
    source_bonus = 1.0
    feature_node_ids = {
        str(node.get("id"))
        for node in nodes
        if node.get("layer") == "code_feature"
    }
    feature_weights = [
        _clamp_graph_weight(relation.get("weight", 0.0))
        for relation in relations
        if relation.get("source") in feature_node_ids
        or relation.get("target") in feature_node_ids
    ]
    feature_overlap = sum(feature_weights) / len(feature_weights) if feature_weights else 0.0
    edge_count = len(relations)
    path_length_penalty = 0.04 * max(edge_count - 2, 0)
    components = {
        "minEdgeWeight": _round_graph_score(min_edge_weight),
        "meanEdgeWeight": _round_graph_score(mean_edge_weight),
        "sourceBonus": _round_graph_score(source_bonus),
        "featureOverlap": _round_graph_score(feature_overlap),
        "pathLengthPenalty": _round_graph_score(path_length_penalty),
    }
    if min_edge_weight <= 0:
        score = 0.0
    else:
        score = _round_graph_score(
            (0.45 * components["minEdgeWeight"])
            + (0.25 * components["meanEdgeWeight"])
            + (0.15 * components["sourceBonus"])
            + (0.15 * components["featureOverlap"])
            - components["pathLengthPenalty"]
        )
    return {
        "strategy": _GRAPH_PATH_SCORING_STRATEGY,
        "score": score,
        "components": components,
    }


def _round_graph_score(value: float) -> float:
    return round(min(max(float(value), 0.0), 1.0), 6)


def _graph_layer_for_node(node_id: str) -> str:
    if node_id.startswith("pattern:"):
        return "pattern"
    if node_id.startswith("code_feature:") or node_id.startswith("code-feature:"):
        return "code_feature"
    if node_id.startswith("source:"):
        return "source"
    if node_id.startswith("chunk:") or ":statement:" in node_id or ":answer:" in node_id:
        return "chunk"
    return "concept"


def _canonical_target_relation_type(
    target: JsonMap,
    target_label: str,
    raw_relation: str | None,
) -> str:
    raw_type = str(raw_relation or "").upper()
    if raw_type in _ALLOWED_GRAPH_RELATION_TYPES:
        return raw_type
    target_layer = str(target.get("layer") or "")
    if target_layer == "pattern":
        return "IMPLEMENTS_PATTERN"
    if target_layer == "code_feature":
        return "HAS_CODE_FEATURE"
    if target_layer == "chunk":
        return "HAS_SECTION"
    if _classify_concept(target_label) == "data_structure":
        return "USES_DATA_STRUCTURE"
    return "MENTIONS_CONCEPT"


def _source_node_id(document: RetrievalDocument) -> str:
    source = _slug(document.source or "source")
    source_id = _slug(document.source_id or document.id)
    return f"source:{source}:{source_id}"


def _source_node_label(document: RetrievalDocument) -> str:
    if document.source and document.source_id:
        return f"{document.source} {document.source_id}"
    return document.source or document.source_id or document.title


def _graph_path_target_id(path: JsonMap) -> str:
    nodes = path.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return ""
    target = nodes[-1]
    if isinstance(target, dict):
        return str(target.get("id") or "")
    return str(target)


def _graph_path_problem_id(path: JsonMap) -> str:
    nodes = path.get("nodes")
    if not isinstance(nodes, list):
        return ""
    string_fallback = ""
    for node in nodes:
        if isinstance(node, dict):
            node_type = str(node.get("type") or node.get("layer") or "").lower()
            node_id = str(node.get("id") or "")
            if node_type == "problem" and node_id:
                return node_id
        else:
            node_id = str(node)
            if not string_fallback and _looks_like_problem_id(node_id):
                string_fallback = node_id
    return string_fallback


def _graph_path_target_concept_id(path: JsonMap) -> str:
    nodes = path.get("nodes")
    if not isinstance(nodes, list):
        return ""
    string_fallback = ""
    semantic_layers = {
        "concept",
        "algorithm",
        "data_structure",
        "technique",
        "pattern",
        "code_feature",
    }
    for node in reversed(nodes):
        if isinstance(node, dict):
            node_type = str(node.get("type") or node.get("layer") or "").lower()
            if node_type in semantic_layers:
                return str(node.get("id") or node.get("label") or "")
        else:
            node_id = str(node)
            if not _looks_like_problem_id(node_id):
                string_fallback = node_id
                break
    return string_fallback


def _graph_path_priority(path: JsonMap) -> tuple[int, int, float, int, str]:
    nodes = path.get("nodes")
    relations = path.get("relations")
    if isinstance(relations, list):
        hop_count = len(relations)
    elif isinstance(nodes, list):
        hop_count = max(len(nodes) - 1, 0)
    else:
        hop_count = 0

    target_label = _graph_path_target_label(path)
    target_id = _graph_path_target_concept_id(path)
    concept_rank = _GRAPH_PATH_CONCEPT_PRIORITY.get(
        _normalize_graph_path_concept_key(target_label),
        _GRAPH_PATH_CONCEPT_PRIORITY.get(
            _normalize_graph_path_concept_key(target_id),
            len(_GRAPH_PATH_CONCEPT_PRIORITY),
        ),
    )
    score = _clamp_graph_weight(path.get("score", _mapping(path.get("pathScoring")).get("score", 0.0)))
    source_rank = 0 if str(path.get("pathSource") or "").lower() == "neo4j" else 1
    target_key = _normalize_graph_path_concept_key(target_label or target_id)
    return (hop_count, concept_rank, -score, source_rank, target_key)


def _prune_graph_paths(
    paths: Sequence[JsonMap],
    target_problem_ids: Sequence[str],
    max_paths_per_problem: int = 2,
) -> tuple[JsonMap, ...]:
    if max_paths_per_problem <= 0:
        return ()

    ordered_problem_ids: list[str] = []
    seen_problem_ids: set[str] = set()
    for problem_id in target_problem_ids:
        problem_key = str(problem_id)
        if problem_key and problem_key not in seen_problem_ids:
            seen_problem_ids.add(problem_key)
            ordered_problem_ids.append(problem_key)

    paths_by_problem: dict[str, dict[str, JsonMap]] = {
        problem_id: {} for problem_id in ordered_problem_ids
    }
    for path in paths:
        problem_id = _graph_path_problem_id(path)
        if problem_id not in paths_by_problem:
            continue
        target_concept_id = _graph_path_target_concept_id(path)
        if not target_concept_id:
            continue
        best_by_concept = paths_by_problem[problem_id]
        existing = best_by_concept.get(target_concept_id)
        if existing is None or _graph_path_priority(path) < _graph_path_priority(existing):
            best_by_concept[target_concept_id] = path

    pruned: list[JsonMap] = []
    for problem_id in ordered_problem_ids:
        problem_paths = sorted(
            paths_by_problem[problem_id].values(),
            key=_graph_path_priority,
        )
        pruned.extend(dict(path) for path in problem_paths[:max_paths_per_problem])
    return tuple(pruned)


def _graph_search_status(
    graph_candidates: Sequence[RetrievalCandidate],
    graph_paths: Sequence[JsonMap],
) -> GraphSearchStatus:
    if graph_candidates:
        return "candidates"
    if graph_paths:
        return "paths_only"
    return "none"


def _graph_path_target_label(path: JsonMap) -> str:
    nodes = path.get("nodes")
    if not isinstance(nodes, list):
        return ""
    semantic_layers = {
        "concept",
        "algorithm",
        "data_structure",
        "technique",
        "pattern",
        "code_feature",
    }
    for node in reversed(nodes):
        if isinstance(node, dict):
            node_type = str(node.get("type") or node.get("layer") or "").lower()
            if node_type in semantic_layers:
                return str(node.get("label") or node.get("id") or "")
        else:
            node_id = str(node)
            if not _looks_like_problem_id(node_id):
                return node_id
    return ""


def _normalize_graph_path_concept_key(value: str) -> str:
    normalized = str(value or "").lower()
    normalized = normalized.removeprefix("concept:")
    normalized = normalized.removeprefix("pattern:")
    normalized = normalized.removeprefix("code_feature:")
    normalized = normalized.replace("-", " ").replace("_", " ")
    return " ".join(normalized.split())


def _looks_like_problem_id(value: str) -> bool:
    lowered = str(value).lower()
    return lowered.startswith(("leetcode-", "uva-"))


def _graph_path_node_label(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("label") or node.get("id") or "")
    return str(node)


def _graph_path_relation_label(relation: Any) -> str:
    if isinstance(relation, dict):
        return str(relation.get("type") or relation.get("relation") or "")
    return str(relation)


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


def _tuple_of_mapping(value: Any) -> tuple[JsonMap, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple(dict(item) for item in value if isinstance(item, dict))
    if isinstance(value, list):
        return tuple(dict(item) for item in value if isinstance(item, dict))
    return ()


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


def _normalize_alias(value: str) -> str:
    return " ".join(_tokens(value))


def _problem_aliases(document: RetrievalDocument) -> set[str]:
    return {
        _normalize_alias(document.title),
        _normalize_alias(f"{document.source} {document.title}"),
        _normalize_alias(f"{document.source} {document.source_id} {document.title}"),
        _normalize_alias(f"{document.source}-{document.source_id} {document.title}"),
        _normalize_alias(f"{document.id} {document.title}"),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def _classify_concept(name: str) -> str:
    lowered = name.lower()
    if lowered in {"bfs", "dfs", "dijkstra", "dynamic programming", "binary search"}:
        return "algorithm"
    if lowered in {"visited array", "visited set", "state tracking"}:
        return "technique"
    if lowered in {"queue", "stack", "heap", "array", "hash map"}:
        return "data_structure"
    return "concept"


def _code_feature_compatibility(
    understanding: QueryUnderstanding | None,
    candidate: RetrievalCandidate,
) -> float:
    if understanding is None or understanding.code_features is None:
        return 0.0
    features = understanding.code_features.features
    if not features:
        return 0.0

    candidate_text = " ".join(
        (
            candidate.title,
            candidate.text,
            " ".join(candidate.concepts),
            candidate.problem_type,
        )
    ).lower()
    matched = 0
    for feature in features:
        terms = _CODE_FEATURE_COMPATIBILITY_TERMS.get(feature, ())
        if any(term in candidate_text for term in terms):
            matched += 1
    return round(matched / len(features), 6)


def _append_unique(values: list[str], item: str) -> None:
    if item not in values:
        values.append(item)


def defaultdict_float() -> dict[str, float]:
    return {}

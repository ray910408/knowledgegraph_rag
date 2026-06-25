from __future__ import annotations

from copy import deepcopy
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Sequence

from ..analysis import detect_input_kind, load_programming_dataset
from ..contracts import RetrievalEvidenceBundle, RetrievalTrace
from ..providers import DeterministicMockEmbeddingProvider, EmbeddingProvider
from ..stores import BM25Store, GraphStore, SearchCandidate, VectorStore


JsonMap = dict[str, Any]
RetrievalMode = Literal["hybrid", "vector", "graph"]
_MAX_STORE_FETCH_ATTEMPTS = 4
_MAX_STORE_FETCH_WINDOW = 100


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


class QueryUnderstandingService:
    def __init__(self, documents: Sequence[RetrievalDocument] = ()) -> None:
        self._documents = tuple(documents)

    def understand(self, query: str) -> QueryUnderstanding:
        normalized = " ".join(query.strip().split())
        lowered = normalized.lower()
        keywords = tuple(dict.fromkeys(_tokens(lowered)))
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
        return QueryUnderstanding(
            original_query=query,
            normalized_query=normalized,
            input_kind=input_kind,
            intent=intent,
            keywords=keywords,
        )


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
        ("concept:bfs", "BFS", "algorithm"),
        ("concept:queue", "Queue", "data_structure"),
        ("concept:visited-array", "Visited Array", "data_structure"),
        ("concept:dynamic-programming", "Dynamic Programming", "algorithm"),
        ("pattern:graph-traversal", "Graph Traversal", "pattern"),
        ("concept:shortest-path", "Shortest Path", "concept"),
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
            linked.append(
                {
                    "entityId": matched_problem.problem_id,
                    "name": matched_problem.title,
                    "type": "problem",
                    "confidence": matched_problem.confidence,
                    "matchKind": matched_problem.match_kind,
                }
            )
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
        if self._bm25_store is not None:
            return _search_and_aggregate_store_candidates(
                lambda window: self._bm25_store.search(
                    understanding.normalized_query,
                    top_k=window,
                ),
                source="bm25",
                top_k=top_k,
            )

        query_terms = set(understanding.keywords)
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
        graph_store: GraphStore | None = None,
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
                    {
                        "nodes": ["input", match["entityId"], document.id],
                        "relations": ["MENTIONS", "REQUIRED_BY"],
                        "score": round(score, 6),
                        "rationale": f"linked {match['name']} to {document.title}",
                        "pathSource": "inferred",
                    }
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
                    str(path["nodes"][-1]): path
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
                }
                for candidate in similar_candidates
            ],
            graph_paths=[dict(path) for path in graph_paths],
            algorithm_evidence=algorithms,
            data_structure_evidence=data_structures,
            pattern_evidence=patterns,
            technique_evidence=techniques,
            matched_problem=matched_problem.to_mapping() if matched_problem else None,
            common_mistakes=[
                "忘記標記 visited。",
                "queue 初始化錯誤，導致起點或距離沒有被正確設定。",
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
        else:
            lines.append("- 無")
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
        lines.extend(["", "圖路徑"])
        for path in evidence["graphPaths"]:
            nodes = [str(node) for node in path.get("nodes", [])]
            relations = [str(relation) for relation in path.get("relations", [])]
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
        graph_store: GraphStore | None = None,
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
        reranked = Reranker().rerank(understanding.normalized_query, fused, top_k=top_k)
        trace = RetrievalTrace(
            query_understanding=understanding.to_mapping(),
            entity_linking=[dict(entity) for entity in linked_entities],
            vector_candidates=[candidate.to_mapping() for candidate in vector_candidates],
            graph_candidates=[candidate.to_mapping() for candidate in graph_result.candidates],
            bm25_candidates=[candidate.to_mapping() for candidate in bm25_candidates],
            fusion_scores=[candidate.to_mapping() for candidate in fused],
            reranker_scores=[candidate.to_mapping() for candidate in reranked],
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
        payload=_candidate_evidence_payload(
            base_payload={},
            metadata={},
            fallback=document,
        ),
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
    is_exact_problem_path = source_node is not None and target_node is not None
    if is_exact_problem_path:
        nodes = ["input", source_node, target_node]
        relations = [
            "EXACT_MATCH",
            raw_relations[-1] if raw_relations else "RELATED_TO",
        ]
        rationale = (
            f"Neo4j returned a path from {document.title} "
            f"to {entity_name}."
        )
    else:
        nodes = ["input", entity_id, document.id]
        relations = ["MENTIONS", "REQUIRED_BY"]
        rationale = f"Neo4j linked {entity_name} to {document.title}."

    return {
        "nodes": nodes,
        "relations": relations,
        "score": _safe_score(path.get("score", 0.0)),
        "rationale": rationale,
        "pathSource": "neo4j",
        "storePath": {
            "nodes": raw_nodes,
            "relations": raw_relations,
        },
    }


def _has_usable_store_path(path: JsonMap) -> bool:
    store_path = path.get("storePath")
    if not isinstance(store_path, dict):
        return False
    return bool(store_path.get("nodes")) and bool(store_path.get("relations"))


def _safe_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return round(score, 6)


def _list_or_tuple_of_str(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value]


def _inferred_problem_paths(document: RetrievalDocument) -> tuple[JsonMap, ...]:
    paths = [
        {
            "nodes": ["input", document.id, f"concept:{_slug(concept)}"],
            "relations": ["EXACT_MATCH", "REQUIRES"],
            "score": 1.0,
            "pathSource": "inferred",
            "rationale": (
                f"Inferred from document concepts; this path was not returned by Neo4j "
                f"for {document.title} and {concept}."
            ),
        }
        for concept in document.concepts
    ]
    if document.problem_type:
        paths.append(
            {
                "nodes": [
                    "input",
                    document.id,
                    f"pattern:{_slug(document.problem_type)}",
                ],
                "relations": ["EXACT_MATCH", "HAS_PATTERN"],
                "score": 1.0,
                "pathSource": "inferred",
                "rationale": (
                    "Inferred from document problem_type; this path was not returned "
                    f"by Neo4j for {document.title} and {document.problem_type}."
                ),
            }
        )
    return tuple(paths)


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


def _append_unique(values: list[str], item: str) -> None:
    if item not in values:
        values.append(item)


def defaultdict_float() -> dict[str, float]:
    return {}

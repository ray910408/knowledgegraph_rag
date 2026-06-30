from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Literal, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .analysis import (
    GRAPH_TRAVERSAL_TYPE,
    analyze_programming_input,
    build_query_id,
    detect_input_kind,
    has_explicit_problem_reference,
    load_programming_dataset,
)
from .demo import build_demo_repositories, recommend_demo_techniques
from .model_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalModelConfig
from .retrieval.pipeline import ContextBuilder, EvidenceBuilder
from .retrieval.runtime import RuntimeRetrieval, add_runtime_debug_trace, build_runtime_retrieval


RecommendationMode = Literal["hybrid", "vector", "graph"]
MAX_ANALYSIS_INPUT_CHARS = 8000
UNSUPPORTED_REASON = "No programming problem, code, concept, or retrieval evidence was detected."


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    problem_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("problemId", "problem_id"),
    )
    problem_text: str | None = Field(
        default=None,
        validation_alias=AliasChoices("problemText", "problem_text"),
    )
    statement: str | None = None
    mode: RecommendationMode = "hybrid"
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        validation_alias=AliasChoices("topK", "top_k"),
    )


class TechniqueResponse(BaseModel):
    id: str
    kind: Literal["algorithm", "data_structure", "pattern"]
    title: str
    score: float
    confidence: Literal["high", "medium", "low"]
    summary: str
    fitSignals: list[str]
    pitfalls: list[str]


class EvidenceNodeResponse(BaseModel):
    id: str
    label: str
    type: str


class EvidenceEdgeResponse(BaseModel):
    from_: str = Field(alias="from")
    to: str
    relation: str
    weight: float


class EvidencePathResponse(BaseModel):
    title: str
    nodes: list[EvidenceNodeResponse]
    edges: list[EvidenceEdgeResponse]


class EvaluationMetricResponse(BaseModel):
    name: str
    vectorOnly: float
    graphOnly: float
    hybrid: float
    note: str


class RecommendationResponse(BaseModel):
    queryId: str
    usedMockData: bool
    recommendations: list[TechniqueResponse]
    evidencePaths: list[EvidencePathResponse]
    evaluation: list[EvaluationMetricResponse]


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    problem_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("problemId", "problem_id"),
    )
    input: str | None = None
    problem_text: str | None = Field(
        default=None,
        validation_alias=AliasChoices("problemText", "problem_text"),
    )
    statement: str | None = None
    code: str | None = None
    mode: RecommendationMode = "hybrid"
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        validation_alias=AliasChoices("topK", "top_k"),
    )


class SimilarProblemResponse(BaseModel):
    source: str
    id: str
    sourceId: str | None = None
    title: str
    reason: str
    sharedConcepts: list[str]
    answerHint: str


class RequiredConceptResponse(BaseModel):
    id: str
    name: str
    kind: str
    description: str


class ProviderDescriptorResponse(BaseModel):
    provider: str
    model: str
    adapter: str | None = None


class RetrievalConfigResponse(BaseModel):
    embeddingModel: str
    rerankerModel: str
    language: str
    embeddingProvider: ProviderDescriptorResponse | None = None
    rerankerProvider: ProviderDescriptorResponse | None = None


class AnalysisEvidenceNodeResponse(BaseModel):
    id: str
    label: str
    type: str


class AnalysisEvidenceEdgeResponse(BaseModel):
    from_: str = Field(alias="from")
    to: str
    relation: str
    weight: float


class AnalysisEvidencePathResponse(BaseModel):
    title: str
    nodes: list[AnalysisEvidenceNodeResponse]
    edges: list[AnalysisEvidenceEdgeResponse]


class MatchedProblemResponse(BaseModel):
    id: str
    title: str
    source: str
    sourceId: str
    matchKind: str
    confidence: float
    score: float
    sharedConcepts: list[str] = Field(default_factory=list)
    problemType: str
    answerHint: str | None = None
    solutionHints: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    constraints: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    queryId: str
    status: Literal["ok", "unsupported"] = "ok"
    abstentionReason: str | None = None
    usedMockData: bool
    inputKind: Literal["problem", "cpp", "python", "unknown"]
    problemType: str
    requiredConcepts: list[RequiredConceptResponse]
    similarProblems: list[SimilarProblemResponse]
    similarityReason: str
    solvingHints: list[str]
    commonMistakes: list[str]
    evidencePaths: list[AnalysisEvidencePathResponse]
    retrievalConfig: RetrievalConfigResponse
    retrievalBackend: Literal["local", "stores"] | None = None
    retrievalTrace: dict[str, Any] | None = None
    evidenceBundle: dict[str, Any] | None = None
    contextPreview: str | None = None
    matchedProblem: MatchedProblemResponse | None = None


class AnalysisInputTooLargeDetail(BaseModel):
    code: Literal["input_too_large"]
    maxLength: int
    actualLength: int


class AnalysisInputTooLargeResponse(BaseModel):
    detail: AnalysisInputTooLargeDetail


def _analysis_paths_from_graph_trace(
    paths: Sequence[object],
) -> list[AnalysisEvidencePathResponse]:
    converted: list[AnalysisEvidencePathResponse] = []
    for path in paths:
        if not isinstance(path, Mapping):
            continue

        raw_nodes = path.get("nodes", [])
        raw_relations = path.get("relations", [])
        if not isinstance(raw_nodes, (list, tuple)) or len(raw_nodes) < 2:
            continue
        if not isinstance(raw_relations, (list, tuple)) or not raw_relations:
            continue

        nodes = [
            node
            for node in (
                _graph_trace_node_response(raw_node)
                for raw_node in raw_nodes
            )
            if node is not None
        ]
        edge_count = len(nodes) - 1
        if edge_count < 1 or len(raw_relations) != edge_count:
            continue

        score = _graph_trace_score(path.get("score", 0.0))
        edges = [
            _graph_trace_edge_response(
                raw_relations[index],
                from_id=nodes[index].id,
                to=nodes[index + 1].id,
                default_weight=score,
            )
            for index in range(edge_count)
        ]
        source = str(path.get("pathSource") or "unknown")
        converted.append(
            AnalysisEvidencePathResponse(
                title=f"Graph path {len(converted) + 1} ({source})",
                nodes=nodes,
                edges=edges,
            )
        )
    return converted


def _graph_trace_node_response(node: object) -> AnalysisEvidenceNodeResponse | None:
    if isinstance(node, Mapping):
        node_id = str(node.get("id") or node.get("label") or "")
        if not node_id:
            return None
        label = str(node.get("label") or node_id)
        node_type = str(node.get("layer") or node.get("type") or _graph_trace_node_type(node_id))
    else:
        node_id = str(node)
        if not node_id:
            return None
        label = node_id
        node_type = _graph_trace_node_type(node_id)
    return AnalysisEvidenceNodeResponse(id=node_id, label=label, type=node_type)


def _graph_trace_edge_response(
    relation: object,
    *,
    from_id: str,
    to: str,
    default_weight: float,
) -> AnalysisEvidenceEdgeResponse:
    if isinstance(relation, Mapping):
        source = str(relation.get("source") or relation.get("from") or from_id)
        target = str(relation.get("target") or relation.get("to") or to)
        relation_type = str(relation.get("type") or relation.get("relation") or "")
        weight = _graph_trace_score(relation.get("weight", default_weight))
    else:
        source = from_id
        target = to
        relation_type = str(relation)
        weight = default_weight
    return AnalysisEvidenceEdgeResponse(
        **{
            "from": source,
            "to": target,
            "relation": relation_type,
            "weight": weight,
        }
    )


def _graph_trace_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return score


def _graph_trace_node_type(node: str) -> str:
    if node == "input":
        return "input"
    if node.startswith("source:"):
        return "source"
    if node.startswith("chunk:"):
        return "chunk"
    if node.startswith("code_feature:") or node.startswith("code-feature:"):
        return "code_feature"
    if node.startswith("concept:"):
        return "concept"
    if node.startswith("pattern:"):
        return "pattern"
    return "problem"


app = FastAPI(title="Explainable Programming GraphRAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def configure_runtime_retrieval() -> None:
    app.state.runtime_retrieval = build_runtime_retrieval()


def _runtime_retrieval() -> RuntimeRetrieval:
    runtime = getattr(app.state, "runtime_retrieval", None)
    if runtime is None:
        runtime = build_runtime_retrieval()
        app.state.runtime_retrieval = runtime
    return runtime


def _retrieval_config_response(
    config: RetrievalModelConfig,
    runtime_retrieval: RuntimeRetrieval,
) -> RetrievalConfigResponse:
    return RetrievalConfigResponse(
        embeddingModel=config.embedding_model,
        rerankerModel=config.reranker_model,
        language=config.language,
        embeddingProvider=_provider_descriptor_response(
            runtime_retrieval.provider_sources.get("embedding")
        ),
        rerankerProvider=_provider_descriptor_response(
            runtime_retrieval.provider_sources.get("reranker")
        ),
    )


def _is_unsupported_analysis(
    text: str,
    input_kind: str,
    pipeline_result: object,
    evidence_mapping: Mapping[str, Any],
    *,
    has_explicit_reference: bool,
) -> bool:
    if has_explicit_reference or detect_input_kind(text) != "unknown":
        return False
    matched_problem = getattr(pipeline_result, "matched_problem", None)
    if matched_problem is not None and getattr(matched_problem, "match_kind", None) != "partial_title":
        return False
    if matched_problem is not None:
        return True
    if input_kind != "unknown":
        return False
    trace = getattr(pipeline_result, "trace").to_mapping()
    has_programming_signal = any(
        trace.get(key)
        for key in ("entityLinking", "graphCandidates")
    )
    has_graph_paths = bool(evidence_mapping.get("graphPaths"))
    return not has_programming_signal and not has_graph_paths


def _is_unsupported_result(result: object) -> bool:
    return (
        getattr(result, "input_kind", None) == "unknown"
        and not getattr(result, "required_concepts", ())
        and not getattr(result, "similar_problems", ())
        and not getattr(result, "evidence_paths", ())
    )


def _has_strong_retrieval_evidence(
    pipeline_result: object,
    evidence_mapping: Mapping[str, Any],
) -> bool:
    if getattr(pipeline_result, "matched_problem", None) is not None:
        return True

    understanding = getattr(pipeline_result, "query_understanding", None)
    concept_seeds = tuple(getattr(understanding, "concept_seeds", ()) or ())
    if not concept_seeds:
        return False
    return bool(evidence_mapping.get("graphPaths") or evidence_mapping.get("similarProblems"))


def _retrieval_problem_type(
    pipeline_result: object,
    evidence_mapping: Mapping[str, Any],
) -> str:
    matched_problem = getattr(pipeline_result, "matched_problem", None)
    if matched_problem is not None:
        candidate = getattr(matched_problem, "candidate", None)
        problem_type = str(getattr(candidate, "problem_type", "") or "")
        if problem_type:
            return _display_problem_type(problem_type)
        matched_mapping = matched_problem.to_mapping()
        return _display_problem_type(str(matched_mapping.get("problemType") or ""))

    for pattern in evidence_mapping.get("patternEvidence") or ():
        pattern_name = str(pattern).strip()
        if pattern_name:
            return _display_problem_type(pattern_name)

    for candidate in getattr(pipeline_result, "reranked_candidates", ()) or ():
        problem_type = str(getattr(candidate, "problem_type", "") or "").strip()
        if problem_type:
            return _display_problem_type(problem_type)
    return ""


def _retrieval_required_concept_responses(
    pipeline_result: object,
    evidence_mapping: Mapping[str, Any],
) -> list[RequiredConceptResponse]:
    concept_names: list[str] = []
    matched_problem = getattr(pipeline_result, "matched_problem", None)
    if matched_problem is not None:
        _append_unique_text(
            concept_names,
            getattr(getattr(matched_problem, "candidate", None), "concepts", ()) or (),
        )

    if not concept_names:
        understanding = getattr(pipeline_result, "query_understanding", None)
        _append_unique_text(concept_names, getattr(understanding, "concept_seeds", ()) or ())
        for candidate in getattr(pipeline_result, "reranked_candidates", ()) or ():
            _append_unique_text(concept_names, getattr(candidate, "concepts", ()) or ())
        for key in (
            "algorithmEvidence",
            "dataStructureEvidence",
            "techniqueEvidence",
            "patternEvidence",
        ):
            _append_unique_text(concept_names, evidence_mapping.get(key) or ())

    return [
        RequiredConceptResponse(
            id=_retrieval_concept_id(name),
            name=name,
            kind=_retrieval_concept_kind(name),
            description=_retrieval_concept_description(name),
        )
        for name in concept_names
    ]


def _retrieval_similarity_reason(
    pipeline_result: object,
    required_concepts: Sequence[RequiredConceptResponse],
) -> str:
    concept_names = [concept.name for concept in required_concepts]
    concept_summary = "、".join(concept_names)
    matched_problem = getattr(pipeline_result, "matched_problem", None)
    if matched_problem is not None:
        source = getattr(matched_problem, "source", "")
        source_id = getattr(matched_problem, "source_id", "")
        title = getattr(matched_problem, "title", "")
        if concept_summary:
            return (
                f"檢索直接命中 {source}-{source_id} {title}，"
                f"並以匹配題目的概念作為依據：{concept_summary}。"
            )
        return f"檢索直接命中 {source}-{source_id} {title}。"

    if concept_summary:
        concept_set = set(concept_names)
        if {"BFS", "Shortest Path"} <= concept_set or (
            "BFS" in concept_set and "Graph Traversal" in concept_set
        ):
            return (
                "檢索證據顯示這些候選都需要在無權圖中找最短步數，"
                f"因此可以用 BFS 找最短步數；證據概念：{concept_summary}。"
            )
        return f"圖譜檢索找到與輸入概念相關的候選題，證據概念：{concept_summary}。"
    return "檢索證據支援此分析結果。"


def _retrieval_solving_hints(evidence_mapping: Mapping[str, Any]) -> list[str]:
    hints: list[str] = []
    matched_problem = evidence_mapping.get("matchedProblem")
    if isinstance(matched_problem, Mapping):
        _append_unique_text(hints, matched_problem.get("solutionHints") or ())
    if hints:
        return hints

    for problem in evidence_mapping.get("similarProblems") or ():
        if isinstance(problem, Mapping):
            _append_unique_text(hints, problem.get("solutionHints") or ())
    return hints


def _append_unique_text(items: list[str], values: object) -> None:
    if isinstance(values, str):
        iterable: Sequence[object] = (values,)
    elif isinstance(values, Sequence):
        iterable = values
    else:
        iterable = ()
    for value in iterable:
        text = str(value).strip()
        if text and text not in items:
            items.append(text)


def _retrieval_concept_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown"


def _retrieval_concept_kind(name: str) -> str:
    lowered = name.lower()
    if lowered in {"bfs", "dfs", "dijkstra", "dynamic programming", "binary search"}:
        return "algorithm"
    if lowered in {"visited array", "visited set", "state tracking"}:
        return "technique"
    if lowered in {"queue", "stack", "heap", "array", "hash map"}:
        return "data_structure"
    if lowered in {"graph traversal", "sliding window"}:
        return "pattern"
    return "concept"


def _retrieval_concept_description(name: str) -> str:
    readable_name = re.sub(r"\s+", " ", name).strip()
    return f"檢索證據指出 {readable_name} 與輸入直接相關。"


def _display_problem_type(problem_type: str) -> str:
    if problem_type == "Graph Traversal":
        return GRAPH_TRAVERSAL_TYPE
    return problem_type


def _unsupported_analysis_response(
    *,
    text: str,
    input_kind: Literal["unknown"],
    retrieval_config: RetrievalConfigResponse,
    retrieval_backend: Literal["local", "stores"] | None,
    retrieval_trace: dict[str, Any],
    evidence_mapping: dict[str, Any],
    debug: bool,
    abstention_reason: str = UNSUPPORTED_REASON,
) -> AnalysisResponse:
    empty_trace = {
        **retrieval_trace,
        "entityLinking": [],
        "vectorCandidates": [],
        "graphCandidates": [],
        "bm25Candidates": [],
        "fusionScores": [],
        "rerankerScores": [],
        "matchedProblem": None,
    }
    empty_evidence = {
        **evidence_mapping,
        "similarProblems": [],
        "graphPaths": [],
        "algorithmEvidence": [],
        "dataStructureEvidence": [],
        "patternEvidence": [],
        "techniqueEvidence": [],
        "commonMistakes": [],
        "matchedProblem": None,
    }
    return AnalysisResponse(
        queryId=build_query_id(text, input_kind),
        status="unsupported",
        abstentionReason=abstention_reason,
        usedMockData=False,
        inputKind=input_kind,
        problemType="",
        requiredConcepts=[],
        similarProblems=[],
        similarityReason="",
        solvingHints=[],
        commonMistakes=[],
        evidencePaths=[],
        retrievalConfig=retrieval_config,
        retrievalBackend=retrieval_backend if debug else None,
        retrievalTrace=empty_trace,
        evidenceBundle=empty_evidence,
        contextPreview=None,
        matchedProblem=None,
    )


def _unsupported_json_response(response: AnalysisResponse) -> JSONResponse:
    payload = response.model_dump(mode="json", by_alias=True, exclude_none=True)
    payload["matchedProblem"] = None
    return JSONResponse(content=payload)


@app.get("/api/v1/health")
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/recommendations", response_model=RecommendationResponse)
@app.post("/api/recommendations", response_model=RecommendationResponse)
def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    graph, _ = build_demo_repositories()
    problem = graph.get_problem(request.problem_id) if request.problem_id else None
    if request.problem_id and problem is None:
        raise HTTPException(status_code=400, detail=f"unknown problemId: {request.problem_id}")

    request_text = request.problem_text or request.statement
    statement = (request_text or (problem.text if problem else "")).strip()
    if not statement:
        raise HTTPException(status_code=400, detail="problemText or statement is required")

    recommendations_ = recommend_demo_techniques(
        graph,
        problem_text=statement,
        top_k=request.top_k,
        mode=request.mode,
    )

    return RecommendationResponse(
        queryId=_query_id(request.problem_id, bool(request_text), request.mode, request.top_k),
        usedMockData=False,
        recommendations=[
            TechniqueResponse(
                id=item.id,
                kind=item.kind,  # type: ignore[arg-type]
                title=item.title,
                score=item.score,
                confidence=item.confidence,  # type: ignore[arg-type]
                summary=item.summary,
                fitSignals=list(item.fit_signals),
                pitfalls=list(item.pitfalls),
            )
            for item in recommendations_
        ],
        evidencePaths=_flatten_paths(graph, recommendations_),
        evaluation=_evaluation_placeholder(),
    )


@app.post(
    "/api/v1/analysis",
    response_model=AnalysisResponse,
    response_model_exclude_none=True,
    responses={
        413: {
            "model": AnalysisInputTooLargeResponse,
            "description": "Analysis input exceeds the supported character limit.",
        }
    },
)
@app.post(
    "/api/analysis",
    response_model=AnalysisResponse,
    response_model_exclude_none=True,
    responses={
        413: {
            "model": AnalysisInputTooLargeResponse,
            "description": "Analysis input exceeds the supported character limit.",
        }
    },
)
def analysis(request: AnalysisRequest, debug: bool = False) -> AnalysisResponse | JSONResponse:
    resolved_problem_text = _analysis_problem_text(request.problem_id)
    raw_text = (
        request.input
        or request.problem_text
        or request.statement
        or request.code
        or resolved_problem_text
        or ""
    )
    if len(raw_text) > MAX_ANALYSIS_INPUT_CHARS:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "input_too_large",
                "maxLength": MAX_ANALYSIS_INPUT_CHARS,
                "actualLength": len(raw_text),
            },
        )
    text = raw_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="input, problemText, statement, or code is required")
    retrieval_query = (
        request.input
        or request.problem_text
        or request.statement
        or request.code
        or request.problem_id
        or resolved_problem_text
        or ""
    ).strip()
    has_explicit_reference = has_explicit_problem_reference(
        retrieval_query,
        load_programming_dataset(),
    )

    runtime_retrieval = _runtime_retrieval()
    pipeline_result = runtime_retrieval.pipeline.run(
        retrieval_query,
        mode=request.mode,
        top_k=request.top_k,
    )
    retrieval_trace = pipeline_result.trace.to_mapping()
    if debug:
        retrieval_trace = add_runtime_debug_trace(
            retrieval_trace,
            runtime_retrieval.candidate_sources,
            runtime_retrieval.provider_sources,
            runtime_retrieval.compatibility_warnings,
        )
    evidence_bundle = EvidenceBuilder().build(
        pipeline_result.reranked_candidates,
        pipeline_result.graph_paths,
        matched_problem=pipeline_result.matched_problem,
    )
    evidence_mapping = evidence_bundle.to_mapping()
    input_kind = pipeline_result.query_understanding.input_kind
    retrieval_config = _retrieval_config_response(
        DEFAULT_RETRIEVAL_CONFIG,
        runtime_retrieval,
    )
    if _is_unsupported_analysis(
        text,
        input_kind,
        pipeline_result,
        evidence_mapping,
        has_explicit_reference=has_explicit_reference,
    ):
        return _unsupported_json_response(
            _unsupported_analysis_response(
                text=text,
                input_kind="unknown",
                retrieval_config=retrieval_config,
                retrieval_backend=runtime_retrieval.backend,
                retrieval_trace=retrieval_trace,
                evidence_mapping=evidence_mapping,
                debug=debug,
            )
        )

    has_strong_retrieval_evidence = _has_strong_retrieval_evidence(
        pipeline_result,
        evidence_mapping,
    )
    result = None
    if not has_strong_retrieval_evidence:
        try:
            result = analyze_programming_input(text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if _is_unsupported_result(result):
            return _unsupported_json_response(
                _unsupported_analysis_response(
                    text=text,
                    input_kind="unknown",
                    retrieval_config=retrieval_config,
                    retrieval_backend=runtime_retrieval.backend,
                    retrieval_trace=retrieval_trace,
                    evidence_mapping=evidence_mapping,
                    debug=debug,
                    abstention_reason=result.similarity_reason,
                )
            )

    retrieval_evidence_paths = _analysis_paths_from_graph_trace(
        evidence_mapping["graphPaths"]
    )
    response_input_kind = (
        "problem" if has_explicit_reference and input_kind == "unknown" else input_kind
    )
    context_preview = ContextBuilder().build(
        pipeline_result.query_understanding,
        evidence_bundle,
    )
    evidence_common_mistakes = list(evidence_mapping.get("commonMistakes") or [])
    if debug:
        retrieval_trace["rawGraphPaths"] = [
            dict(path) for path in pipeline_result.raw_graph_paths
        ]
    matched_problem_ids = (
        {
            pipeline_result.matched_problem.problem_id,
            pipeline_result.matched_problem.source_id,
        }
        if pipeline_result.matched_problem is not None
        else set()
    )
    if has_strong_retrieval_evidence:
        problem_type = _retrieval_problem_type(pipeline_result, evidence_mapping)
        required_concepts = _retrieval_required_concept_responses(
            pipeline_result,
            evidence_mapping,
        )
        similarity_reason = _retrieval_similarity_reason(
            pipeline_result,
            required_concepts,
        )
        solving_hints = _retrieval_solving_hints(evidence_mapping)
        used_mock_data = False
    else:
        assert result is not None
        problem_type = result.problem_type
        required_concepts = [
            RequiredConceptResponse(
                id=concept.id,
                name=concept.name,
                kind=concept.kind,
                description=concept.description,
            )
            for concept in result.required_concepts
        ]
        similarity_reason = result.similarity_reason
        solving_hints = list(result.solving_hints)
        used_mock_data = result.used_mock_data
    common_mistakes = evidence_common_mistakes or (
        list(result.common_mistakes) if result is not None else []
    )

    return AnalysisResponse(
        queryId=build_query_id(text, response_input_kind),
        usedMockData=used_mock_data,
        inputKind=response_input_kind,
        problemType=problem_type,
        requiredConcepts=required_concepts,
        similarProblems=_similar_problem_responses_from_candidates(
            pipeline_result.reranked_candidates,
            matched_problem_ids=matched_problem_ids,
        ),
        similarityReason=similarity_reason,
        solvingHints=solving_hints,
        commonMistakes=common_mistakes,
        evidencePaths=retrieval_evidence_paths,
        retrievalConfig=retrieval_config,
        retrievalBackend=runtime_retrieval.backend if debug else None,
        retrievalTrace=retrieval_trace,
        evidenceBundle=evidence_mapping,
        contextPreview=context_preview if debug else None,
        matchedProblem=(
            pipeline_result.matched_problem.to_mapping()
            if pipeline_result.matched_problem is not None
            else None
        ),
    )


def _provider_descriptor_response(
    descriptor: dict[str, Any] | None,
) -> ProviderDescriptorResponse | None:
    if descriptor is None:
        return None
    return ProviderDescriptorResponse(**descriptor)


def _similar_problem_responses_from_candidates(
    candidates: Sequence[object],
    *,
    matched_problem_ids: set[str],
) -> list[SimilarProblemResponse]:
    responses: list[SimilarProblemResponse] = []
    for candidate in candidates:
        candidate_id = str(getattr(candidate, "id", ""))
        payload = getattr(candidate, "payload", {})
        payload_map = payload if isinstance(payload, Mapping) else {}
        source_id_value = payload_map.get("sourceId") or payload_map.get("source_id")
        source_id = str(source_id_value) if source_id_value else None
        if candidate_id in matched_problem_ids or source_id in matched_problem_ids:
            continue
        concepts = [str(concept) for concept in getattr(candidate, "concepts", ())]
        responses.append(
            SimilarProblemResponse(
                source=str(payload_map.get("documentSource") or getattr(candidate, "source", "")),
                id=candidate_id,
                sourceId=source_id,
                title=str(getattr(candidate, "title", "")),
                reason=_similar_problem_reason(concepts),
                sharedConcepts=concepts,
                answerHint=str(payload_map.get("answer") or ""),
            )
        )
    return responses


def _similar_problem_reason(concepts: Sequence[str]) -> str:
    if concepts:
        return f"所選檢索模式將此題列為最終候選，並共享這些概念：{'、'.join(concepts)}。"
    return "所選檢索模式將此題列為最終重排序候選。"


def _analysis_problem_text(problem_id: str | None) -> str | None:
    if problem_id is None:
        return None
    for problem in load_programming_dataset():
        if problem.id == problem_id or problem.source_id == problem_id:
            return problem.statement
    raise HTTPException(status_code=404, detail=f"unknown problemId: {problem_id}")


def _flatten_paths(graph, recommendations_) -> list[EvidencePathResponse]:
    responses: list[EvidencePathResponse] = []
    for recommendation in recommendations_:
        for path_index, path in enumerate(recommendation.evidence_paths, start=1):
            responses.append(
                EvidencePathResponse(
                    title=f"{recommendation.title} evidence {path_index}",
                    nodes=[
                        EvidenceNodeResponse(
                            id=node_id,
                            label=_node_label(graph, node_id),
                            type=_node_type(graph, node_id),
                        )
                        for node_id in path.nodes
                    ],
                    edges=[
                        EvidenceEdgeResponse(
                            **{
                                "from": path.nodes[index],
                                "to": path.nodes[index + 1],
                                "relation": relation,
                                "weight": path.score,
                            }
                        )
                        for index, relation in enumerate(path.relations)
                        if index + 1 < len(path.nodes)
                    ],
                )
            )
    return responses


def _query_id(
    problem_id: str | None,
    has_request_text: bool,
    mode: RecommendationMode,
    top_k: int,
) -> str:
    source = problem_id or "demo"
    if problem_id and has_request_text:
        source = f"{problem_id}-with-text"
    return f"{source}-{mode}-{top_k}"


def _node_label(graph, node_id: str) -> str:
    if node_id == "query":
        return "Input problem"
    concept = graph.get_concept(node_id)
    if concept:
        return concept.name
    problem = graph.get_problem(node_id)
    if problem:
        return problem.title
    return node_id


def _node_type(graph, node_id: str) -> str:
    if node_id == "query":
        return "problem"
    concept = graph.get_concept(node_id)
    if concept:
        return concept.kind
    if graph.get_problem(node_id):
        return "problem"
    return "concept"


def _evaluation_placeholder() -> list[EvaluationMetricResponse]:
    return [
        EvaluationMetricResponse(
            name="Concept recall",
            vectorOnly=0.72,
            graphOnly=0.65,
            hybrid=0.84,
            note="Demo placeholder until the frozen CPE/LeetCode set is imported.",
        ),
        EvaluationMetricResponse(
            name="Explainability",
            vectorOnly=0.38,
            graphOnly=0.81,
            hybrid=0.88,
            note="Graph and hybrid modes expose typed evidence paths.",
        ),
        EvaluationMetricResponse(
            name="Noise control",
            vectorOnly=0.58,
            graphOnly=0.70,
            hybrid=0.79,
            note="Hybrid reranking is expected to reduce unrelated matches.",
        ),
    ]

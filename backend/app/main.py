from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .analysis import analyze_programming_input, build_query_id, load_programming_dataset
from .demo import build_demo_repositories, recommend_demo_techniques
from .retrieval.pipeline import ContextBuilder, EvidenceBuilder
from .retrieval.runtime import RuntimeRetrieval, add_runtime_debug_trace, build_runtime_retrieval


RecommendationMode = Literal["hybrid", "vector", "graph"]


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


class SimilarProblemResponse(BaseModel):
    source: str
    id: str
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

        nodes = [str(node) for node in raw_nodes]
        relations = [str(relation) for relation in raw_relations]
        edge_count = len(nodes) - 1
        if len(relations) != edge_count:
            continue

        try:
            score = float(path.get("score", 0.0))
        except (TypeError, ValueError, OverflowError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        source = str(path.get("pathSource") or "unknown")
        converted.append(
            AnalysisEvidencePathResponse(
                title=f"Graph path {len(converted) + 1} ({source})",
                nodes=[
                    AnalysisEvidenceNodeResponse(
                        id=node,
                        label=node,
                        type=_graph_trace_node_type(node),
                    )
                    for node in nodes
                ],
                edges=[
                    AnalysisEvidenceEdgeResponse(
                        **{
                            "from": nodes[index],
                            "to": nodes[index + 1],
                            "relation": relations[index],
                            "weight": score,
                        }
                    )
                    for index in range(edge_count)
                ],
            )
        )
    return converted


def _graph_trace_node_type(node: str) -> str:
    if node == "input":
        return "input"
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


@app.post("/api/v1/analysis", response_model=AnalysisResponse, response_model_exclude_none=True)
@app.post("/api/analysis", response_model=AnalysisResponse, response_model_exclude_none=True)
def analysis(request: AnalysisRequest, debug: bool = False) -> AnalysisResponse:
    resolved_problem_text = _analysis_problem_text(request.problem_id)
    text = (
        request.input
        or request.problem_text
        or request.statement
        or request.code
        or resolved_problem_text
        or ""
    ).strip()
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

    try:
        result = analyze_programming_input(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    runtime_retrieval = _runtime_retrieval()
    pipeline_result = runtime_retrieval.pipeline.run(retrieval_query, top_k=5)
    retrieval_trace = pipeline_result.trace.to_mapping()
    if debug:
        retrieval_trace = add_runtime_debug_trace(
            retrieval_trace,
            runtime_retrieval.candidate_sources,
            runtime_retrieval.provider_sources,
        )
    evidence_bundle = EvidenceBuilder().build(
        pipeline_result.reranked_candidates,
        pipeline_result.graph_paths,
        matched_problem=pipeline_result.matched_problem,
    )
    evidence_mapping = evidence_bundle.to_mapping()
    retrieval_evidence_paths = _analysis_paths_from_graph_trace(
        evidence_mapping["graphPaths"]
    )
    context_preview = ContextBuilder().build(
        pipeline_result.query_understanding,
        evidence_bundle,
    )
    input_kind = pipeline_result.query_understanding.input_kind
    matched_problem_ids = (
        {
            pipeline_result.matched_problem.problem_id,
            pipeline_result.matched_problem.source_id,
        }
        if pipeline_result.matched_problem is not None
        else set()
    )

    return AnalysisResponse(
        queryId=build_query_id(text, input_kind),
        usedMockData=result.used_mock_data,
        inputKind=input_kind,
        problemType=result.problem_type,
        requiredConcepts=[
            RequiredConceptResponse(
                id=concept.id,
                name=concept.name,
                kind=concept.kind,
                description=concept.description,
            )
            for concept in result.required_concepts
        ],
        similarProblems=[
            SimilarProblemResponse(
                source=problem.source,
                id=problem.id,
                title=problem.title,
                reason=problem.reason,
                sharedConcepts=list(problem.shared_concepts),
                answerHint=problem.answer_hint,
            )
            for problem in result.similar_problems
            if problem.id not in matched_problem_ids
        ],
        similarityReason=result.similarity_reason,
        solvingHints=list(result.solving_hints),
        commonMistakes=list(result.common_mistakes),
        evidencePaths=retrieval_evidence_paths
        or [
            AnalysisEvidencePathResponse(
                title=path.title,
                nodes=[
                    AnalysisEvidenceNodeResponse(
                        id=node.id,
                        label=node.label,
                        type=node.type,
                    )
                    for node in path.nodes
                ],
                edges=[
                    AnalysisEvidenceEdgeResponse(
                        **{
                            "from": edge.from_id,
                            "to": edge.to,
                            "relation": edge.relation,
                            "weight": edge.weight,
                        }
                    )
                    for edge in path.edges
                ],
            )
            for path in result.evidence_paths
        ],
        retrievalConfig=RetrievalConfigResponse(
            embeddingModel=result.retrieval_config.embedding_model,
            rerankerModel=result.retrieval_config.reranker_model,
            language=result.retrieval_config.language,
            embeddingProvider=_provider_descriptor_response(
                runtime_retrieval.provider_sources.get("embedding")
            ),
            rerankerProvider=_provider_descriptor_response(
                runtime_retrieval.provider_sources.get("reranker")
            ),
        ),
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

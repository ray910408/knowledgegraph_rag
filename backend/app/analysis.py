from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .model_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalModelConfig
from .query_language import detect_traversal_signal_labels


InputKind = Literal["problem", "cpp", "python", "unknown"]

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "programming_problems.json"

GRAPH_TRAVERSAL_TYPE = "\u5716\u8ad6\u904d\u6b77\uff08Graph Traversal\uff09"
SIMILARITY_REASON = (
    "\u90fd\u9700\u8981\u5728\u7121\u6b0a\u5716\u4e2d\u627e\u6700\u77ed\u6b65\u6578\uff0c"
    "\u56e0\u6b64\u53ef\u4ee5\u7528 BFS \u627e\u6700\u77ed\u6b65\u6578\u3002"
)
SOLVING_HINTS = (
    "\u5148\u5efa\u5716\uff0c\u518d BFS\u3002",
    "\u628a\u8d77\u9ede\u653e\u5165 Queue\uff0c\u4e26\u5728\u5165\u968a\u6642\u6a19\u8a18 visited\u3002",
)
COMMON_MISTAKES = (
    "\u5fd8\u8a18\u6a19\u8a18 visited\u3002",
    "queue \u521d\u59cb\u5316\u932f\u8aa4\uff0c\u5c0e\u81f4\u8d77\u9ede\u6216\u8ddd\u96e2\u6c92\u6709\u88ab\u6b63\u78ba\u8a2d\u5b9a\u3002",
)


@dataclass(frozen=True)
class ProgrammingProblem:
    id: str
    title: str
    source: str
    source_id: str
    problem_type: str
    concepts: tuple[str, ...]
    statement: str
    answer: str
    solution_hints: tuple[str, ...]
    metadata: dict[str, str]


@dataclass(frozen=True)
class RequiredConcept:
    id: str
    name: str
    kind: str
    description: str


@dataclass(frozen=True)
class SimilarProblemAnalysis:
    source: str
    id: str
    title: str
    reason: str
    shared_concepts: tuple[str, ...]
    answer_hint: str


@dataclass(frozen=True)
class EvidenceNodeAnalysis:
    id: str
    label: str
    type: str


@dataclass(frozen=True)
class EvidenceEdgeAnalysis:
    from_id: str
    to: str
    relation: str
    weight: float


@dataclass(frozen=True)
class EvidencePathAnalysis:
    title: str
    nodes: tuple[EvidenceNodeAnalysis, ...]
    edges: tuple[EvidenceEdgeAnalysis, ...]


@dataclass(frozen=True)
class AnalysisResult:
    query_id: str
    used_mock_data: bool
    input_kind: InputKind
    problem_type: str
    required_concepts: tuple[RequiredConcept, ...]
    similar_problems: tuple[SimilarProblemAnalysis, ...]
    similarity_reason: str
    solving_hints: tuple[str, ...]
    common_mistakes: tuple[str, ...]
    evidence_paths: tuple[EvidencePathAnalysis, ...]
    retrieval_config: RetrievalModelConfig


def analyze_programming_input(user_input: str) -> AnalysisResult:
    text = user_input.strip()
    if not text:
        raise ValueError("input is required")

    input_kind = detect_input_kind(text)
    dataset = load_programming_dataset()
    graph_examples = find_graph_traversal_examples(dataset)
    signals = detect_graph_traversal_signals(text)
    has_supported_graph_signal = has_supported_graph_traversal_signals(text)
    concepts = graph_traversal_concepts()

    if (
        input_kind == "unknown"
        and not has_supported_graph_signal
        and not has_explicit_problem_reference(text, dataset)
    ):
        return AnalysisResult(
            query_id=build_query_id(text, input_kind),
            used_mock_data=False,
            input_kind=input_kind,
            problem_type="不支援的問題",
            required_concepts=(),
            similar_problems=(),
            similarity_reason="This input is outside the supported graph traversal analysis scope.",
            solving_hints=(),
            common_mistakes=(),
            evidence_paths=(),
            retrieval_config=DEFAULT_RETRIEVAL_CONFIG,
        )

    return AnalysisResult(
        query_id=build_query_id(text, input_kind),
        used_mock_data=False,
        input_kind=input_kind,
        problem_type=GRAPH_TRAVERSAL_TYPE,
        required_concepts=concepts,
        similar_problems=tuple(to_similar_problem(problem) for problem in graph_examples[:3]),
        similarity_reason=SIMILARITY_REASON,
        solving_hints=SOLVING_HINTS,
        common_mistakes=COMMON_MISTAKES,
        evidence_paths=build_evidence_paths(signals, concepts),
        retrieval_config=DEFAULT_RETRIEVAL_CONFIG,
    )


def load_programming_dataset(path: Path = DATASET_PATH) -> tuple[ProgrammingProblem, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    problems = raw.get("problems", [])
    return tuple(_parse_problem(item) for item in problems)


def find_graph_traversal_examples(
    dataset: tuple[ProgrammingProblem, ...],
) -> tuple[ProgrammingProblem, ...]:
    examples = [
        problem
        for problem in dataset
        if problem.problem_type == "Graph Traversal"
        and {"BFS", "Queue", "Visited Array"}.issubset(set(problem.concepts))
    ]
    examples.sort(key=lambda problem: (problem.source != "UVa", problem.source, problem.source_id))
    return tuple(examples)


def has_explicit_problem_reference(
    text: str,
    dataset: tuple[ProgrammingProblem, ...],
) -> bool:
    normalized = _normalize_problem_reference(text)
    if not normalized:
        return False
    for problem in dataset:
        exact_aliases = {
            problem.id,
            problem.source_id,
            problem.title,
            f"{problem.source} {problem.source_id}",
            f"{problem.source}-{problem.source_id}",
            f"{problem.source} {problem.source_id} {problem.title}",
            f"{problem.source}-{problem.source_id} {problem.title}",
            f"{problem.id} {problem.title}",
        }
        normalized_exact_aliases = {
            _normalize_problem_reference(alias)
            for alias in exact_aliases
        }
        if normalized in normalized_exact_aliases:
            return True
        identifier_aliases = {
            _normalize_problem_reference(problem.id),
            _normalize_problem_reference(problem.source_id),
            _normalize_problem_reference(f"{problem.source} {problem.source_id}"),
            _normalize_problem_reference(f"{problem.source}-{problem.source_id}"),
        }
        if any(
            alias and f" {alias} " in f" {normalized} "
            for alias in identifier_aliases
        ):
            return True
    return False


def _normalize_problem_reference(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def graph_traversal_concepts() -> tuple[RequiredConcept, ...]:
    return (
        RequiredConcept(
            id="bfs",
            name="BFS",
            kind="algorithm",
            description="\u5728\u7121\u6b0a\u5716\u4e2d\u9010\u5c64\u64f4\u5c55\uff0c\u7528\u4f86\u627e\u6700\u77ed\u6b65\u6578\u3002",
        ),
        RequiredConcept(
            id="queue",
            name="Queue",
            kind="data_structure",
            description="\u7dad\u6301 BFS \u5f85\u8655\u7406\u7684\u7bc0\u9ede\u9806\u5e8f\u3002",
        ),
        RequiredConcept(
            id="visited-array",
            name="Visited Array",
            kind="data_structure",
            description="\u8a18\u9304\u5df2\u7d93\u5165\u968a\u6216\u8655\u7406\u904e\u7684\u7bc0\u9ede\uff0c\u907f\u514d\u91cd\u8907\u64f4\u5c55\u3002",
        ),
    )


def to_similar_problem(problem: ProgrammingProblem) -> SimilarProblemAnalysis:
    return SimilarProblemAnalysis(
        source=problem.source,
        id=problem.source_id,
        title=problem.title,
        reason=SIMILARITY_REASON,
        shared_concepts=problem.concepts,
        answer_hint=problem.answer,
    )


def detect_input_kind(text: str) -> InputKind:
    lowered = text.lower()
    cpp_markers = (
        "#include",
        "int main",
        "using namespace",
        "std::queue",
        "queue<",
        "vector<",
        "cin >>",
        "cout <<",
    )
    python_markers = (
        "from collections import deque",
        "deque(",
        "def ",
        "q.popleft",
        ".append(",
        "visited =",
    )
    if any(marker in lowered for marker in cpp_markers):
        return "cpp"
    if any(marker in lowered for marker in python_markers):
        return "python"
    if has_supported_graph_traversal_signals(text):
        return "problem"
    return "unknown"


def has_supported_graph_traversal_signals(text: str) -> bool:
    signals = set(detect_graph_traversal_signals(text))
    if "BFS" in signals or "Unweighted shortest path" in signals:
        return True
    if "Graph" in signals and {"Queue", "Visited Array"} & signals:
        return True
    return False


def detect_graph_traversal_signals(text: str) -> tuple[str, ...]:
    return detect_traversal_signal_labels(text)


def build_query_id(text: str, input_kind: InputKind) -> str:
    normalized = re.sub(r"\s+", "-", text.lower())[:32].strip("-")
    normalized = re.sub(r"[^a-z0-9_-]+", "", normalized) or "input"
    return f"analysis-{input_kind}-{normalized}"


def build_evidence_paths(
    signals: tuple[str, ...],
    concepts: tuple[RequiredConcept, ...],
) -> tuple[EvidencePathAnalysis, ...]:
    if not signals:
        signals = ("Graph", "Unweighted shortest path")
    nodes = (
        EvidenceNodeAnalysis(id="input", label="\u8f38\u5165\u5167\u5bb9", type="problem"),
        EvidenceNodeAnalysis(id="graph-traversal", label=GRAPH_TRAVERSAL_TYPE, type="pattern"),
        *(
            EvidenceNodeAnalysis(id=concept.id, label=concept.name, type=concept.kind)
            for concept in concepts
        ),
    )
    edges = (
        EvidenceEdgeAnalysis(
            from_id="input",
            to="graph-traversal",
            relation="\u7b26\u5408\u8f38\u5165\u8a0a\u865f",
            weight=1.0,
        ),
        *(
            EvidenceEdgeAnalysis(
                from_id="graph-traversal",
                to=concept.id,
                relation="\u9700\u8981\u89c0\u5ff5",
                weight=1.0,
            )
            for concept in concepts
        ),
    )
    return (
        EvidencePathAnalysis(
            title="\u5716\u8ad6\u904d\u6b77 BFS \u5206\u6790\u8b49\u64da",
            nodes=nodes,
            edges=edges,
        ),
    )


def _parse_problem(item: dict[str, Any]) -> ProgrammingProblem:
    source = str(item.get("source", item.get("platform", "")))
    return ProgrammingProblem(
        id=str(item["id"]),
        title=str(item["title"]),
        source=source,
        source_id=str(item["sourceId"]),
        problem_type=str(item["problemType"]),
        concepts=tuple(str(value) for value in item.get("concepts", [])),
        statement=str(item["statement"]),
        answer=str(item["answer"]),
        solution_hints=tuple(str(value) for value in item.get("solutionHints", [])),
        metadata={str(key): str(value) for key, value in item.get("metadata", {}).items()},
    )

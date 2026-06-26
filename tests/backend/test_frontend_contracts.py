import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
API_TS = REPO_ROOT / "frontend" / "src" / "api.ts"
APP_TSX = REPO_ROOT / "frontend" / "src" / "App.tsx"


def _function_body(source: str, marker: str) -> str:
    start = source.index(marker)
    open_brace = source.index("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"could not find function body for {marker}")


def test_fetch_analysis_does_not_return_mock_fallback_on_live_failures():
    source = API_TS.read_text(encoding="utf-8")
    body = _function_body(source, "export async function fetchAnalysis")

    assert "return mockFallback(request)" not in body
    assert "throw new Error" in body
    assert "analysis API is unavailable" in body


def test_normalize_response_does_not_convert_invalid_payload_to_mock_success():
    source = API_TS.read_text(encoding="utf-8")
    body = _function_body(source, "function normalizeResponse")

    assert "return mockFallback(request)" not in body
    assert "invalid analysis API payload" in source
    assert "invalidPayload(" in body
    assert re.search(
        r'status\s*===\s*"ok"\s*&&\s*\(\s*'
        r"requiredConcepts\.length\s*===\s*0\s*\|\|\s*\(\s*"
        r"similarProblems\.length\s*===\s*0\s*&&\s*!matchedProblem\s*"
        r"\)\s*\)",
        body,
    )


def test_fetch_analysis_wraps_malformed_success_json_as_invalid_payload():
    source = API_TS.read_text(encoding="utf-8")
    body = _function_body(source, "export async function fetchAnalysis")

    assert "return normalizeResponse(await response.json(), request)" not in body
    assert re.search(
        r"try\s*{\s*payload\s*=\s*await\s+response\.json\(\);\s*}\s*catch",
        body,
        flags=re.DOTALL,
    )
    assert "invalidPayload(" in body
    assert "return normalizeResponse(payload, request)" in body


def test_normalize_response_rejects_explicit_invalid_status_values():
    source = API_TS.read_text(encoding="utf-8")
    body = _function_body(source, "function normalizeAnalysisStatus")

    assert re.search(r"value\s*===\s*null\s*\|\|\s*value\s*===\s*undefined", body)
    assert re.search(r'value\s*===\s*"ok"\s*\|\|\s*value\s*===\s*"unsupported"', body)
    assert 'invalidPayload("status must be ok or unsupported")' in body


def test_normalize_trace_preserves_compatibility_warnings_for_debug_json():
    source = API_TS.read_text(encoding="utf-8")
    body = _function_body(source, "function normalizeTrace")

    assert "compatibilityWarnings" in body
    assert "compatibility_warnings" in body


def test_app_clears_previous_result_when_new_request_starts():
    source = APP_TSX.read_text(encoding="utf-8")
    body = _function_body(source, "async function handleAnalyze")

    loading_index = body.index("setIsLoading(true);")
    error_index = body.index("setError(null);")
    clear_index = body.index("setResponse(null);", error_index)
    fetch_index = body.index("fetchAnalysis(", clear_index)

    assert loading_index < error_index < clear_index < fetch_index
    assert "latestRequestId.current" in body
    assert "requestId === latestRequestId.current" in body


def test_app_preserves_raw_input_for_server_side_limit_contract():
    source = APP_TSX.read_text(encoding="utf-8")
    body = _function_body(source, "async function handleAnalyze")

    assert "const maxAnalysisInputChars = 8000;" in source
    assert "const rawInput = inputText;" in body
    assert "const trimmedInput = rawInput.trim();" in body
    assert "trimmedInput.length === 0 && rawInput.length <= maxAnalysisInputChars" in body
    assert "fetchAnalysis({ inputText: rawInput, mode, topK, debug })" in body
    assert "fetchAnalysis({ inputText: trimmedInput" not in body


def test_app_renders_unsupported_response_as_abstention_only():
    source = APP_TSX.read_text(encoding="utf-8")

    branch_index = source.index('if (response.status === "unsupported")')
    normal_trace_index = source.index("const trace = response.retrievalTrace", branch_index)
    unsupported_branch = source[branch_index:normal_trace_index]

    assert branch_index < normal_trace_index
    assert "response.abstentionReason" in unsupported_branch
    assert "MatchedProblemPanel" not in unsupported_branch
    assert "RetrievalPanel" not in unsupported_branch
    assert "EvidenceBundlePanel" not in unsupported_branch
    assert 'response?.status !== "unsupported"' in source


def test_app_renders_graph_path_scoring_metadata():
    source = APP_TSX.read_text(encoding="utf-8")
    start = source.index("function GraphPathsPanel")
    end = source.index("function ContextPanel", start)
    body = source[start:end]

    assert "graphPathOperationLabel(path.graphPathOperation)" in body
    assert "path.pathScoring?.strategy" in body
    assert "formatScore(path.pathScoring?.score)" in body
    assert "formatPathScoringComponents(path.pathScoring?.components)" in body
    assert "function formatPathScoringComponents" in source


def test_frontend_fallback_graph_path_uses_current_scoring_contract():
    source = API_TS.read_text(encoding="utf-8")

    assert "graphPathOperation: \"exact_expansion\"" in source
    assert "score: 0.85" in source
    assert "sourceBonus: 1" in source
    assert "featureOverlap: 0" in source
    assert "pathLengthPenalty: 0" in source
    assert "sourceBonus: 0.85" not in source
    assert "pathLengthPenalty: 1" not in source

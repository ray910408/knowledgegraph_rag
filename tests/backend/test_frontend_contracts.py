import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
API_TS = REPO_ROOT / "frontend" / "src" / "api.ts"
APP_TSX = REPO_ROOT / "frontend" / "src" / "App.tsx"
TYPES_TS = REPO_ROOT / "frontend" / "src" / "types.ts"


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


def test_frontend_trace_contract_supports_multilingual_query_understanding_fields():
    api_source = API_TS.read_text(encoding="utf-8")
    types_source = TYPES_TS.read_text(encoding="utf-8")
    app_source = APP_TSX.read_text(encoding="utf-8")
    body = _function_body(api_source, "function normalizeTrace")

    for needle in (
        "queryLanguage",
        "exactTerms",
        "lowWeightTerms",
        "conceptSeeds",
        "expandedTerms",
        "queryVariants",
    ):
        assert needle in body
        assert needle in types_source

    assert "QueryUnderstandingPanel" in app_source
    assert "conceptSeeds" in app_source
    assert "expandedTerms" in app_source


def test_frontend_trace_contract_supports_graph_search_status():
    api_source = API_TS.read_text(encoding="utf-8")
    types_source = TYPES_TS.read_text(encoding="utf-8")
    app_source = APP_TSX.read_text(encoding="utf-8")
    normalize_trace_body = _function_body(api_source, "function normalizeTrace")
    retrieval_trace_body = _function_body(types_source, "export interface RetrievalTrace")
    start = app_source.index("function RetrievalPanel")
    end = app_source.index("function FusionPanel", start)
    retrieval_panel_body = app_source[start:end]

    assert re.search(
        r'type\s+GraphSearchStatus\s*=\s*"none"\s*\|\s*"candidates"\s*\|\s*'
        r'"paths_only"\s*\|\s*\(string\s*&\s*\{\s*\}\s*\)\s*;',
        types_source,
    )
    assert re.search(
        r"^\s*graphSearchStatus\?:\s*GraphSearchStatus;\s*$",
        retrieval_trace_body,
        flags=re.MULTILINE,
    )
    assert (
        'firstPresent(record, ["graphSearchStatus", "graph_search_status"])'
        in normalize_trace_body
    )
    assert 'trace?.graphSearchStatus === "paths_only"' in retrieval_panel_body
    assert "有圖路徑證據，但沒有候選題目。" in retrieval_panel_body


def test_frontend_accepts_technique_concept_kind():
    api_source = API_TS.read_text(encoding="utf-8")
    types_source = TYPES_TS.read_text(encoding="utf-8")
    app_source = APP_TSX.read_text(encoding="utf-8")
    concept_kind_body = _function_body(api_source, "function normalizeConceptKind")
    node_type_body = _function_body(api_source, "function normalizeNodeType")

    assert '"technique"' in types_source
    assert 'value === "technique"' in concept_kind_body
    assert 'value === "technique"' in node_type_body
    assert 'case "technique"' in app_source
    assert "技巧" in app_source
    assert re.search(
        r'name:\s*"Visited Array",\s*kind:\s*"technique"',
        api_source,
        flags=re.DOTALL,
    )


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
    assert "response.similarProblems" not in unsupported_branch
    assert "response.evidenceBundle" not in unsupported_branch
    assert "MatchedProblemPanel" not in unsupported_branch
    assert "RetrievalPanel" not in unsupported_branch
    assert "EvidenceBundlePanel" not in unsupported_branch
    assert 'response?.status !== "unsupported"' in source


def test_app_does_not_special_case_dp_uva_437_or_sample_hint_filtering():
    source = APP_TSX.read_text(encoding="utf-8")

    assert 'inputText === "DP"' not in source
    assert '"uva-437"' not in source
    assert "The Tower of Babylon" not in source
    assert "solutionHints.filter" not in source


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


def test_touched_files_use_utf8_without_bom_and_keep_zh_hant_retrieval_copy():
    app_bytes = APP_TSX.read_bytes()
    test_bytes = Path(__file__).read_bytes()
    source = app_bytes.decode("utf-8")

    assert not app_bytes.startswith(b"\xef\xbb\xbf")
    assert not test_bytes.startswith(b"\xef\xbb\xbf")
    assert "<p className=\"eyebrow\">模型設定</p>" in source
    assert "<p className=\"eyebrow\">圖證據</p>" in source
    assert "<p className=\"muted\">尚無證據路徑。</p>" in source
    assert "<p className=\"eyebrow\">圖路徑追蹤</p>" in source
    assert "<p className=\"muted\">沒有圖路徑。</p>" in source
    assert "<dt>節點</dt>" in source
    assert "<dt>關係</dt>" in source
    assert "<dt>依據</dt>" in source
    assert "<dt>分數</dt>" in source
    assert "<dt>來源</dt>" in source
    assert "<p className=\"muted\">Debug mode 關閉。</p>" in source
    assert 'rightTitle="常見錯誤"' in source


def test_retrieval_panel_limits_hybrid_status_to_hybrid_like_traces():
    app_source = APP_TSX.read_text(encoding="utf-8")
    api_source = API_TS.read_text(encoding="utf-8")
    start = app_source.index("function RetrievalPanel")
    end = app_source.index("function FusionPanel", start)
    retrieval_body = app_source[start:end]

    assert 'label: "Hybrid"' in app_source
    assert 'const [mode, setMode] = useState<RetrievalMode>("hybrid");' in app_source
    assert 'title="三路檢索"' in retrieval_body
    assert 'title="向量搜尋 / Qdrant"' in retrieval_body
    assert 'title="圖搜尋 / Neo4j"' in retrieval_body
    assert 'title="BM25 關鍵字搜尋"' in retrieval_body
    assert 'value === "hybrid"' not in api_source
    assert 'status must be ok or unsupported' in api_source


def test_matched_problem_panel_surfaces_problem_specific_common_mistakes():
    source = APP_TSX.read_text(encoding="utf-8")

    unsupported_branch_index = source.index('if (response.status === "unsupported")')
    matched_problem_index = source.index("<MatchedProblemPanel problem={matchedProblem} />")
    common_mistakes_index = source.index('rightTitle="常見錯誤"', matched_problem_index)
    evidence_common_mistakes_index = source.index(
        '<EvidenceList title="常見錯誤" items={evidence.commonMistakes} />'
    )

    assert unsupported_branch_index < matched_problem_index < common_mistakes_index
    assert 'rightItems={response.commonMistakes}' in source
    assert evidence_common_mistakes_index > matched_problem_index
    assert 'rightTitle="Common mistakes"' not in source


def test_frontend_fallback_graph_path_uses_current_scoring_contract():
    source = API_TS.read_text(encoding="utf-8")

    assert "graphPathOperation: \"exact_expansion\"" in source
    assert "score: 0.85" in source
    assert "sourceBonus: 1" in source
    assert "featureOverlap: 0" in source
    assert "pathLengthPenalty: 0" in source
    assert "sourceBonus: 0.85" not in source
    assert "pathLengthPenalty: 1" not in source

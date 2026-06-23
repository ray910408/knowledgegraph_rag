from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _powershell_array_items(script: str, assignment_name: str) -> list[str]:
    lines = script.splitlines()
    assignment_start = f"{assignment_name} = @("
    for line_index, line in enumerate(lines):
        if line.strip() != assignment_start:
            continue

        items: list[str] = []
        for item_line in lines[line_index + 1 :]:
            stripped = item_line.strip()
            if stripped == ")":
                return items
            if stripped:
                items.append(stripped.removesuffix(","))
        break

    raise AssertionError(f"PowerShell array not found: {assignment_name}")


def _backend_job_param_items(script: str) -> list[str]:
    lines = script.splitlines()
    backend_job_start = next(
        (
            line_index
            for line_index, line in enumerate(lines)
            if line.strip().startswith('$backendJob = Start-Job -Name "knowledgegraph-rag-backend"')
        ),
        None,
    )
    if backend_job_start is None:
        raise AssertionError("backend Start-Job block not found")

    param_start = next(
        (
            line_index
            for line_index in range(backend_job_start, len(lines))
            if lines[line_index].strip() == "param("
        ),
        None,
    )
    if param_start is None:
        raise AssertionError("backend Start-Job param block not found")

    items: list[str] = []
    for item_line in lines[param_start + 1 :]:
        stripped = item_line.strip()
        if stripped == ")":
            return items
        if stripped:
            items.append(stripped.removesuffix(","))

    raise AssertionError("backend Start-Job param block was not closed")


def _assert_immediately_after(items: list[str], *, previous: str, expected: str) -> None:
    assert previous in items
    previous_index = items.index(previous)
    assert previous_index + 1 < len(items)
    assert items[previous_index + 1] == expected


def test_quick_start_passes_custom_backend_origin_to_vite_proxy():
    # Regression: quick-start -BackendPort used to leave Vite proxy on port 8000.
    # Found by /qa on 2026-06-23.
    script = (REPO_ROOT / "scripts" / "quick-start.ps1").read_text(encoding="utf-8")
    vite_config = (REPO_ROOT / "frontend" / "vite.config.ts").read_text(encoding="utf-8")

    assert '$BackendOrigin = "http://127.0.0.1:$BackendPort"' in script
    assert "$env:VITE_BACKEND_ORIGIN = $BackendOriginValue" in script
    assert "Receive-Job -Job $job -ErrorAction SilentlyContinue" in script
    assert "Receive-Job -Job $job -Keep" not in script
    assert "--reload" not in script
    assert "VITE_BACKEND_ORIGIN" in vite_config
    assert '|| "http://127.0.0.1:8000"' in vite_config


def test_quick_start_sets_processed_problem_path_for_stores_mode():
    script = (REPO_ROOT / "scripts" / "quick-start.ps1").read_text(encoding="utf-8")
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "PROCESSED_PROBLEMS_PATH=data/processed/problems.json" in env_example
    assert '$ProcessedProblemsPath = Join-Path $ProcessedDataDir "problems.json"' in script
    assert 'Write-Step "Processed problems: $ProcessedProblemsPath"' in script
    assert "$ProcessedProblemsPath" in script
    assert "$ProcessedProblemsValue" in script
    assert "$env:PROCESSED_PROBLEMS_PATH = $ProcessedProblemsValue" in script


def test_quick_start_backend_job_keeps_processed_problem_argument_order():
    script = (REPO_ROOT / "scripts" / "quick-start.ps1").read_text(encoding="utf-8")

    backend_args = _powershell_array_items(script, "$backendArgs")
    backend_params = _backend_job_param_items(script)

    assert backend_args == [
        "$RepoRoot",
        "$BackendPort",
        "$PythonCommand",
        "$RetrievalBackend",
        "$QdrantUrl",
        "$QdrantCollection",
        "$Neo4jUri",
        "$Neo4jUser",
        "$Neo4jPassword",
        "$Bm25IndexPath",
        "$ProcessedProblemsPath",
    ]
    assert backend_params == [
        "$Root",
        "$Port",
        "$Python",
        "$Backend",
        "$QdrantUrlValue",
        "$QdrantCollectionValue",
        "$Neo4jUriValue",
        "$Neo4jUserValue",
        "$Neo4jPasswordValue",
        "$Bm25IndexValue",
        "$ProcessedProblemsValue",
    ]
    _assert_immediately_after(
        backend_args,
        previous="$Bm25IndexPath",
        expected="$ProcessedProblemsPath",
    )
    _assert_immediately_after(
        backend_params,
        previous="$Bm25IndexValue",
        expected="$ProcessedProblemsValue",
    )

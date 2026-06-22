[CmdletBinding()]
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [switch]$SkipInstall,
    [switch]$Check,
    [switch]$Stores,
    [switch]$SkipDocker,
    [switch]$SkipIngestion
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$FrontendDir = Join-Path $RepoRoot "frontend"
$RawDataDir = Join-Path $RepoRoot "data\raw"
$ProcessedDataDir = Join-Path $RepoRoot "data\processed"
$RetrievalBackend = if ($Stores) { "stores" } else { "local" }
$QdrantUrl = "http://localhost:6333"
$QdrantCollection = "programming_chunks"
$Neo4jUri = "bolt://localhost:7687"
$Neo4jUser = "neo4j"
$Neo4jPassword = "password"
$Bm25IndexPath = Join-Path $ProcessedDataDir "bm25_index.json"

function Write-Step {
    param([string]$Message)
    Write-Host "[quick-start] $Message"
}

function Resolve-RequiredCommand {
    param(
        [string]$Name,
        [string[]]$FallbackNames = @()
    )

    $names = @($Name) + $FallbackNames
    foreach ($candidate in $names) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    throw "Required command not found: $Name"
}

function Wait-HttpEndpoint {
    param(
        [string]$Url,
        [int]$Attempts = 30
    )

    for ($index = 1; $index -le $Attempts; $index++) {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 | Out-Null
            return
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }

    throw "Timed out waiting for $Url"
}

$PythonCommand = Resolve-RequiredCommand -Name "python"
$NpmCommand = Resolve-RequiredCommand -Name "npm.cmd" -FallbackNames @("npm")
$DockerCommand = $null
if ($Stores -and -not $SkipDocker) {
    $DockerCommand = Resolve-RequiredCommand -Name "docker"
}

if (-not (Test-Path (Join-Path $RepoRoot "backend\app\main.py"))) {
    throw "Backend entry not found: backend\app\main.py"
}

if (-not (Test-Path (Join-Path $FrontendDir "package.json"))) {
    throw "Frontend package.json not found"
}

if ($Check) {
    Write-Step "Workspace: $RepoRoot"
    Write-Step "Python: $PythonCommand"
    Write-Step "npm: $NpmCommand"
    Write-Step "Retrieval backend: $RetrievalBackend"
    Write-Step "Backend URL: http://127.0.0.1:$BackendPort"
    Write-Step "Frontend URL: http://127.0.0.1:$FrontendPort"
    if ($Stores) {
        if ($DockerCommand) {
            Write-Step "Docker: $DockerCommand"
        }
        else {
            Write-Step "Docker: skipped"
        }
        Write-Step "Qdrant URL: $QdrantUrl"
        Write-Step "Qdrant collection: $QdrantCollection"
        Write-Step "Neo4j URI: $Neo4jUri"
        Write-Step "BM25 index: $Bm25IndexPath"
    }
    Write-Step "Check complete. No services started."
    exit 0
}

if (-not $SkipInstall -and -not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
    Write-Step "Installing frontend dependencies"
    Push-Location $FrontendDir
    try {
        & $NpmCommand install
    }
    finally {
        Pop-Location
    }
}

if ($Stores) {
    if (-not $SkipDocker) {
        Write-Step "Starting Neo4j and Qdrant with Docker Compose"
        Push-Location $RepoRoot
        try {
            & $DockerCommand compose up -d neo4j qdrant
        }
        finally {
            Pop-Location
        }
        Write-Step "Waiting for Qdrant"
        Wait-HttpEndpoint -Url $QdrantUrl
        Write-Step "Waiting for Neo4j browser endpoint"
        Wait-HttpEndpoint -Url "http://localhost:7474"
    }

    if (-not $SkipIngestion) {
        Write-Step "Running ingestion into Qdrant, Neo4j, and BM25 index"
        Push-Location $RepoRoot
        try {
            & $PythonCommand -m backend.app.ingestion build `
                --input $RawDataDir `
                --processed $ProcessedDataDir `
                --target all
        }
        finally {
            Pop-Location
        }
    }
}

Write-Step "Starting backend and frontend. Press Ctrl+C to stop."
Write-Step "Retrieval backend: $RetrievalBackend"
Write-Step "Backend: http://127.0.0.1:$BackendPort"
Write-Step "Frontend: http://127.0.0.1:$FrontendPort"

$backendArgs = @(
    $RepoRoot,
    $BackendPort,
    $PythonCommand,
    $RetrievalBackend,
    $QdrantUrl,
    $QdrantCollection,
    $Neo4jUri,
    $Neo4jUser,
    $Neo4jPassword,
    $Bm25IndexPath
)
$backendJob = Start-Job -Name "knowledgegraph-rag-backend" -ScriptBlock {
    param(
        $Root,
        $Port,
        $Python,
        $Backend,
        $QdrantUrlValue,
        $QdrantCollectionValue,
        $Neo4jUriValue,
        $Neo4jUserValue,
        $Neo4jPasswordValue,
        $Bm25IndexValue
    )
    Set-Location $Root
    $env:PYTHONPATH = $Root
    $env:RETRIEVAL_BACKEND = $Backend
    $env:QDRANT_URL = $QdrantUrlValue
    $env:QDRANT_COLLECTION = $QdrantCollectionValue
    $env:NEO4J_URI = $Neo4jUriValue
    $env:NEO4J_USER = $Neo4jUserValue
    $env:NEO4J_PASSWORD = $Neo4jPasswordValue
    $env:BM25_INDEX_PATH = $Bm25IndexValue
    & $Python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port $Port
} -ArgumentList $backendArgs

$frontendJob = Start-Job -Name "knowledgegraph-rag-frontend" -ScriptBlock {
    param($Frontend, $Port, $Npm)
    Set-Location $Frontend
    & $Npm run dev -- --host 127.0.0.1 --port $Port
} -ArgumentList $FrontendDir, $FrontendPort, $NpmCommand

try {
    while ($true) {
        foreach ($job in @($backendJob, $frontendJob)) {
            Receive-Job -Job $job -Keep | ForEach-Object { Write-Host $_ }
            if ($job.State -in @("Failed", "Stopped", "Completed")) {
                throw "Service stopped: $($job.Name) ($($job.State))"
            }
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Step "Stopping services"
    Stop-Job -Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $backendJob, $frontendJob -Force -ErrorAction SilentlyContinue
}

[CmdletBinding()]
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [switch]$SkipInstall,
    [switch]$Check
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$FrontendDir = Join-Path $RepoRoot "frontend"

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

$PythonCommand = Resolve-RequiredCommand -Name "python"
$NpmCommand = Resolve-RequiredCommand -Name "npm.cmd" -FallbackNames @("npm")

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
    Write-Step "Backend URL: http://127.0.0.1:$BackendPort"
    Write-Step "Frontend URL: http://127.0.0.1:$FrontendPort"
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

Write-Step "Starting backend and frontend. Press Ctrl+C to stop."
Write-Step "Backend: http://127.0.0.1:$BackendPort"
Write-Step "Frontend: http://127.0.0.1:$FrontendPort"

$backendJob = Start-Job -Name "knowledgegraph-rag-backend" -ScriptBlock {
    param($Root, $Port, $Python)
    Set-Location $Root
    $env:PYTHONPATH = $Root
    & $Python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port $Port
} -ArgumentList $RepoRoot, $BackendPort, $PythonCommand

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

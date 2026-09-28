param(
    [switch]$OpenBrowser,
    [string]$CheckpointPath,
    [switch]$Production
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$viewerRoot = Join-Path $projectRoot "viewer"
$runtimeRoot = Join-Path $projectRoot "outputs\runtime"
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$checkpointPointer = Join-Path $runtimeRoot "showcase_checkpoint.txt"
$fallbackCheckpoint = Join-Path $projectRoot "experiments\20260828T175822Z_multidomain_surface_pilot\checkpoint_best_landscape.pt"

if ([string]::IsNullOrWhiteSpace($CheckpointPath) -and (Test-Path -LiteralPath $checkpointPointer)) {
    $CheckpointPath = (Get-Content -LiteralPath $checkpointPointer -Raw).Trim()
}
if ([string]::IsNullOrWhiteSpace($CheckpointPath)) {
    $CheckpointPath = $fallbackCheckpoint
}
if (-not [System.IO.Path]::IsPathRooted($CheckpointPath)) {
    $CheckpointPath = Join-Path $projectRoot $CheckpointPath
}
$CheckpointPath = [System.IO.Path]::GetFullPath($CheckpointPath)

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Monocular Surface Reconstruction Python environment is missing: $pythonPath"
}
if (-not (Test-Path -LiteralPath $CheckpointPath)) {
    throw "Monocular Surface Reconstruction checkpoint is missing: $CheckpointPath"
}

New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null

function Test-LocalPort {
    param([int]$Port)
    return $null -ne (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
}

function Wait-ForUrl {
    param(
        [string]$Url,
        [int]$Seconds = 45
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) {
                return $true
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

if (-not (Test-LocalPort -Port 8000)) {
    $api = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @(
            "scripts\serve_api.py",
            "--checkpoint", $CheckpointPath,
            "--device", "cuda",
            "--host", "127.0.0.1",
            "--port", "8000"
        ) `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput (Join-Path $runtimeRoot "api_stdout.log") `
        -RedirectStandardError (Join-Path $runtimeRoot "api_stderr.log") `
        -WindowStyle Hidden `
        -PassThru
    $api.Id | Out-File -LiteralPath (Join-Path $runtimeRoot "api.pid") -Encoding ascii
    Write-Host "Started inference API (PID $($api.Id))."
}
else {
    Write-Host "Inference API is already listening on port 8000."
}

if (-not (Wait-ForUrl -Url "http://127.0.0.1:8000/api/health" -Seconds 45)) {
    throw "Inference API did not become healthy. Check outputs\runtime\api_stderr.log."
}

if (-not (Test-LocalPort -Port 3000)) {
    $npmCommand = Get-Command npm.cmd -ErrorAction Stop
    $viewerScript = if ($Production) { "start" } else { "dev" }
    $viewer = Start-Process `
        -FilePath $npmCommand.Source `
        -ArgumentList @("run", $viewerScript) `
        -WorkingDirectory $viewerRoot `
        -RedirectStandardOutput (Join-Path $runtimeRoot "viewer_stdout.log") `
        -RedirectStandardError (Join-Path $runtimeRoot "viewer_stderr.log") `
        -WindowStyle Hidden `
        -PassThru
    $viewer.Id | Out-File -LiteralPath (Join-Path $runtimeRoot "viewer.pid") -Encoding ascii
    Write-Host "Started $viewerScript viewer (PID $($viewer.Id))."
}
else {
    Write-Host "Viewer is already listening on port 3000."
}

if (-not (Wait-ForUrl -Url "http://localhost:3000/" -Seconds 45)) {
    throw "Viewer did not become healthy. Check outputs\runtime\viewer_stderr.log."
}

Write-Host "Monocular Surface Reconstruction is ready: http://localhost:3000/" -ForegroundColor Green
Write-Host "Inference checkpoint: $CheckpointPath" -ForegroundColor Green
Write-Host "Viewer mode: $(if ($Production) { 'production' } else { 'development' })" -ForegroundColor Green

if ($OpenBrowser) {
    Start-Process "http://localhost:3000/"
}

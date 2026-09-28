param(
    [string]$TargetRoot = "D:\MSRData\GAMUS",
    [ValidateRange(1, 8)]
    [int]$MaxWorkers = 4
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python is missing: $python"
}

Write-Host "Stage 1/2: complete balanced GAMUS triplets first" -ForegroundColor Cyan
& $python (Join-Path $projectRoot "scripts\data\download_gamus_priority.py") `
    --target $TargetRoot `
    --max-workers $MaxWorkers
if ($LASTEXITCODE -ne 0) {
    throw "Priority GAMUS download failed with exit code $LASTEXITCODE"
}

Write-Host "Stage 2/2: resume the full official GAMUS snapshot" -ForegroundColor Cyan
& (Join-Path $projectRoot "scripts\download_gamus.ps1") `
    -TargetRoot $TargetRoot `
    -MaxWorkers $MaxWorkers
if ($LASTEXITCODE -ne 0) {
    throw "Full GAMUS download failed with exit code $LASTEXITCODE"
}

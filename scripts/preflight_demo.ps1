$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $projectRoot "outputs\runtime"
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$relativeModel = Join-Path $projectRoot "models\foundation\depth-anything-v2-small-hf"
$pointerPath = Join-Path $runtimeRoot "showcase_checkpoint.txt"
$fallbackCheckpoint = Join-Path $projectRoot "experiments\20260828T175822Z_multidomain_surface_pilot\checkpoint_best_landscape.pt"
$failures = [System.Collections.Generic.List[string]]::new()

function Test-RequiredPath {
    param(
        [string]$Label,
        [string]$Path
    )
    if (Test-Path -LiteralPath $Path) {
        Write-Host "[PASS] $Label" -ForegroundColor Green
    }
    else {
        Write-Host "[FAIL] $Label - $Path" -ForegroundColor Red
        $failures.Add($Label)
    }
}

Write-Host "Monocular Surface Reconstruction showcase preflight" -ForegroundColor Cyan
Test-RequiredPath -Label "Python environment" -Path $pythonPath
Test-RequiredPath -Label "Relative-depth foundation model" -Path $relativeModel

$checkpointPath = $fallbackCheckpoint
if (Test-Path -LiteralPath $pointerPath) {
    $checkpointPath = (Get-Content -LiteralPath $pointerPath -Raw).Trim()
    if (-not [System.IO.Path]::IsPathRooted($checkpointPath)) {
        $checkpointPath = Join-Path $projectRoot $checkpointPath
    }
}
Test-RequiredPath -Label "Selected showcase checkpoint" -Path $checkpointPath

$npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($null -eq $npmCommand) {
    Write-Host "[FAIL] Node/npm runtime" -ForegroundColor Red
    $failures.Add("Node/npm runtime")
}
else {
    Write-Host "[PASS] Node/npm runtime" -ForegroundColor Green
}

if (Test-Path -LiteralPath $pythonPath) {
    # Use Python single-quoted literals so Windows/PowerShell does not strip
    # the embedded quotes while forwarding the one-line program to `-c`.
    $cudaProbe = "import torch; ok=torch.cuda.is_available(); print(str(ok)+'|'+(torch.cuda.get_device_name(0) if ok else 'none'))"
    $cudaStatus = & $pythonPath -c $cudaProbe
    if ($LASTEXITCODE -eq 0 -and $cudaStatus -like "True|*") {
        Write-Host "[PASS] CUDA inference - $($cudaStatus.Split('|')[1])" -ForegroundColor Green
    }
    else {
        Write-Host "[FAIL] CUDA inference unavailable" -ForegroundColor Red
        $failures.Add("CUDA inference")
    }
}

foreach ($port in 3000, 8000) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    if ($null -eq $listener) {
        Write-Host "[PASS] Port $port is available" -ForegroundColor Green
    }
    else {
        Write-Host "[INFO] Port $port is already listening; stop the previous prototype before a clean demo."
    }
}

if ($failures.Count -gt 0) {
    Write-Host "Preflight failed: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}

Write-Host "Preflight passed. Build and launch with:" -ForegroundColor Green
Write-Host "npm --prefix viewer run build"
Write-Host ".\scripts\start_prototype.ps1 -Production -OpenBrowser"

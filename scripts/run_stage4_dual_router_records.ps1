$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot "configs\stage4_dual_router_records.yaml"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "stage4_dual_router_records_v3.log"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

Set-Location $projectRoot
if (-not (Test-Path -LiteralPath $python)) {
    throw "Monocular Surface Reconstruction virtual-environment Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Stage-4 rich-record configuration is missing: $configPath"
}

$activeGpuWork = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.ProcessId -ne $PID -and
    $_.CommandLine -match '(train_multidomain|train_scene_router|train_stage4_dual_router|build_stage3_scene_router_records|build_stage4_dual_router_records|evaluate_routed_surface)\.py'
}
if ($activeGpuWork) {
    $activeIds = ($activeGpuWork | Select-Object -ExpandProperty ProcessId) -join ', '
    throw "Another Monocular Surface Reconstruction GPU process is active (PID: $activeIds)"
}

"[$(Get-Date -Format o)] START/RESUME Stage-4 dual-router rich-record generation" |
    Tee-Object -FilePath $logPath -Append
"[$(Get-Date -Format o)] Expected: 3,347 records (2,208 fit; 1,139 calibration); protected app pointer stays unchanged" |
    Tee-Object -FilePath $logPath -Append

# Required for deterministic CUDA matrix multiplication and safe resume.
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $python -u scripts\build_stage4_dual_router_records.py `
    --config configs\stage4_dual_router_records.yaml 2>&1 |
    Tee-Object -FilePath $logPath -Append
$commandExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($commandExitCode -ne 0) {
    "[$(Get-Date -Format o)] FAILED Stage-4 rich-record generation (exit $commandExitCode)" |
        Tee-Object -FilePath $logPath -Append
    throw "Stage-4 dual-router rich-record generation failed with exit code $commandExitCode"
}

"[$(Get-Date -Format o)] COMPLETE Stage-4 rich records; protected app pointer untouched" |
    Tee-Object -FilePath $logPath -Append

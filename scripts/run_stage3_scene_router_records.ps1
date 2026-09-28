$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot "configs\stage3_scene_router_records.yaml"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "stage3_scene_router_records.log"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

Set-Location $projectRoot
if (-not (Test-Path -LiteralPath $python)) {
    throw "Monocular Surface Reconstruction virtual-environment Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Stage-3 router record configuration is missing: $configPath"
}

$activeGpuWork = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.ProcessId -ne $PID -and
    $_.CommandLine -match '(train_multidomain|build_stage3_scene_router_records)\.py'
}
if ($activeGpuWork) {
    $activeIds = ($activeGpuWork | Select-Object -ExpandProperty ProcessId) -join ', '
    throw "Another Monocular Surface Reconstruction GPU training/record process is active (PID: $activeIds)"
}

"[$(Get-Date -Format o)] START/RESUME Stage-3 scene-router record generation" |
    Tee-Object -FilePath $logPath -Append
# Required by PyTorch deterministic CUDA matmul. The value is inherited by
# Python and makes a resumed batch numerically reproducible.
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $python -u scripts\build_stage3_scene_router_records.py `
    --config configs\stage3_scene_router_records.yaml 2>&1 |
    Tee-Object -FilePath $logPath -Append
$commandExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($commandExitCode -ne 0) {
    "[$(Get-Date -Format o)] FAILED Stage-3 record generation (exit $commandExitCode)" |
        Tee-Object -FilePath $logPath -Append
    throw "Stage-3 scene-router record generation failed with exit code $commandExitCode"
}

"[$(Get-Date -Format o)] COMPLETE Stage-3 scene-router records; app pointer untouched" |
    Tee-Object -FilePath $logPath -Append

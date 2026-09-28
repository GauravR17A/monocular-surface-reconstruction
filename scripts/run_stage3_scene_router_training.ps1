$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot "configs\stage3_scene_router_training.yaml"
$recordsPath = Join-Path $projectRoot "outputs\stage3_scene_router_records\records.jsonl"
$provenancePath = Join-Path $projectRoot "outputs\stage3_scene_router_records\provenance.json"
$completionPath = Join-Path $projectRoot "outputs\stage3_scene_router_records\completion.json"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "stage3_scene_router_training.log"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

Set-Location $projectRoot
foreach ($requiredPath in @($python, $configPath, $recordsPath, $provenancePath, $completionPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Stage-3 router training is not ready; required file is missing: $requiredPath"
    }
}

$completion = Get-Content -LiteralPath $completionPath -Raw | ConvertFrom-Json
if ($completion.complete -ne $true) {
    throw "Stage-3 scene-router record generation has not completed"
}

$activeWork = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.ProcessId -ne $PID -and
    $_.CommandLine -match '(train_scene_router|build_stage3_scene_router_records)\.py'
}
if ($activeWork) {
    $activeIds = ($activeWork | Select-Object -ExpandProperty ProcessId) -join ', '
    throw "Another Stage-3 router process is active (PID: $activeIds)"
}

"[$(Get-Date -Format o)] START guarded Stage-3 scene-router training" |
    Tee-Object -FilePath $logPath -Append
# Required by PyTorch deterministic CUDA matmul.
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $python -u scripts\train_scene_router.py `
    --config configs\stage3_scene_router_training.yaml 2>&1 |
    Tee-Object -FilePath $logPath -Append
$commandExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($commandExitCode -ne 0) {
    "[$(Get-Date -Format o)] FAILED Stage-3 router training (exit $commandExitCode)" |
        Tee-Object -FilePath $logPath -Append
    throw "Stage-3 scene-router training failed with exit code $commandExitCode"
}

"[$(Get-Date -Format o)] COMPLETE guarded router artifact; app pointer untouched" |
    Tee-Object -FilePath $logPath -Append

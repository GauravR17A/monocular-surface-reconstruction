$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot "configs\stage4_dual_router_training.yaml"
$recordRoot = Join-Path $projectRoot "outputs\stage4_dual_router_records_v3"
$provenancePath = Join-Path $recordRoot "provenance.json"
$completionPath = Join-Path $recordRoot "completion.json"
$recordsPath = Join-Path $recordRoot "records.jsonl"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "stage4_dual_router_training.log"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

Set-Location $projectRoot
foreach ($requiredPath in @(
    $python,
    $configPath,
    $provenancePath,
    $completionPath,
    $recordsPath
)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Stage-4 training is not ready; required file is missing: $requiredPath"
    }
}

$completion = Get-Content -LiteralPath $completionPath -Raw | ConvertFrom-Json
if (
    $completion.complete -ne $true -or
    $completion.test_splits_excluded -ne $true -or
    $completion.app_pointer_changed -ne $false
) {
    throw "Stage-4 rich-record completion proof is not eligible for training"
}

$activeWork = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.ProcessId -ne $PID -and
    $_.CommandLine -match '(train_stage4_dual_router|build_stage4_dual_router_records)\.py'
}
if ($activeWork) {
    $activeIds = ($activeWork | Select-Object -ExpandProperty ProcessId) -join ', '
    throw "Another Stage-4 process is active (PID: $activeIds)"
}

"[$(Get-Date -Format o)] START guarded Stage-4 dual-router training" |
    Tee-Object -FilePath $logPath -Append
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $python -u scripts\train_stage4_dual_router.py `
    --config configs\stage4_dual_router_training.yaml 2>&1 |
    Tee-Object -FilePath $logPath -Append
$commandExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($commandExitCode -ne 0) {
    "[$(Get-Date -Format o)] FAILED Stage-4 training (exit $commandExitCode)" |
        Tee-Object -FilePath $logPath -Append
    throw "Stage-4 dual-router training failed with exit code $commandExitCode"
}

"[$(Get-Date -Format o)] COMPLETE Stage-4 artifact; app pointer untouched" |
    Tee-Object -FilePath $logPath -Append

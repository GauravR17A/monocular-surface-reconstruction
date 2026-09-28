$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot "configs\stage3b_scene_domain_router_training.yaml"
$recordsPath = Join-Path $projectRoot "outputs\stage3_scene_router_records\records.jsonl"
$provenancePath = Join-Path $projectRoot "outputs\stage3_scene_router_records\provenance.json"
$completionPath = Join-Path $projectRoot "outputs\stage3_scene_router_records\completion.json"
$outputDirectory = Join-Path $projectRoot "experiments\stage3_scene_domain_router_guarded"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "stage3b_scene_domain_router_training.log"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

Set-Location $projectRoot
foreach ($requiredPath in @($python, $configPath, $recordsPath, $provenancePath, $completionPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Stage-3b domain-router training is not ready; missing: $requiredPath"
    }
}
$completion = Get-Content -LiteralPath $completionPath -Raw | ConvertFrom-Json
if ($completion.complete -ne $true -or [int]$completion.descriptor_size -ne 128) {
    throw "Stage-3b requires a complete authenticated 128-D record store"
}
foreach ($artifactName in @("scene_domain_router.pt", "scene_domain_router_report.json")) {
    $artifactPath = Join-Path $outputDirectory $artifactName
    if (Test-Path -LiteralPath $artifactPath) {
        throw "Refusing to overwrite the versioned Stage-3b artifact: $artifactPath"
    }
}

$activeWork = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.ProcessId -ne $PID -and
    $_.CommandLine -match '(train_scene_router|build_stage3_scene_router_records)\.py'
}
if ($activeWork) {
    $activeIds = ($activeWork | Select-Object -ExpandProperty ProcessId) -join ', '
    throw "Another Stage-3 process is active (PID: $activeIds)"
}

"[$(Get-Date -Format o)] START Stage-3b source/domain-router training" |
    Tee-Object -FilePath $logPath -Append
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $python -u scripts\train_scene_router.py `
    --config configs\stage3b_scene_domain_router_training.yaml 2>&1 |
    Tee-Object -FilePath $logPath -Append
$commandExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($commandExitCode -ne 0) {
    "[$(Get-Date -Format o)] FAILED Stage-3b domain router (exit $commandExitCode)" |
        Tee-Object -FilePath $logPath -Append
    throw "Stage-3b domain-router training failed with exit code $commandExitCode"
}

"[$(Get-Date -Format o)] COMPLETE guarded Stage-3b artifact; v1/app untouched" |
    Tee-Object -FilePath $logPath -Append

param(
    [string]$Config = "configs/stage4b_semantic_ood_router.yaml"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $projectRoot "scripts\train_stage4b_semantic_ood_router.py"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "stage4b_semantic_ood_router.log"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Monocular Surface Reconstruction virtual-environment Python was not found: $python"
}
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

Push-Location $projectRoot
try {
    "[$(Get-Date -Format o)] START isolated Stage-4b semantic/OOD router" |
        Tee-Object -FilePath $logPath -Append
    "[$(Get-Date -Format o)] 3 geographic folds x 5 seeds; height hard-disabled; no official test or app promotion" |
        Tee-Object -FilePath $logPath -Append
    & $python $trainer --config $Config 2>&1 |
        Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Stage-4b trainer failed with exit code $LASTEXITCODE"
    }
    "[$(Get-Date -Format o)] COMPLETE Stage-4b offline artifact; app pointer untouched" |
        Tee-Object -FilePath $logPath -Append
}
finally {
    Pop-Location
}

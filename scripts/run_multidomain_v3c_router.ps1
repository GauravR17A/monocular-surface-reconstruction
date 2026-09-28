$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "multidomain_v3c_router.log"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Invoke-LoggedPython {
    param([string[]]$Arguments)
    "[$(Get-Date -Format o)] python $($Arguments -join ' ')" | Tee-Object -FilePath $logPath -Append
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $commandExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($commandExitCode -ne 0) {
        throw "Monocular Surface Reconstruction v3c command failed with exit code $commandExitCode"
    }
}

Set-Location $projectRoot
$latestCheckpoint = Get-ChildItem -Path (Join-Path $projectRoot "experiments") `
    -Directory -Filter "*_multidomain_surface_v3c_router" |
    Sort-Object LastWriteTime -Descending |
    ForEach-Object { Join-Path $_.FullName "checkpoint_latest.pt" } |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1

if ($latestCheckpoint) {
    $savedConfig = Join-Path (Split-Path -Parent $latestCheckpoint) "config.yaml"
    Invoke-LoggedPython -Arguments @(
        "scripts\train_multidomain.py",
        "--config", $savedConfig,
        "--resume", $latestCheckpoint
    )
}
else {
    Invoke-LoggedPython -Arguments @(
        "scripts\train_multidomain.py",
        "--config", "configs\multidomain_surface_v3c_router.yaml"
    )
}

"[$(Get-Date -Format o)] Multidomain v3c router refinement complete" |
    Tee-Object -FilePath $logPath -Append

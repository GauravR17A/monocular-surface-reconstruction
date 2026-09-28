$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "gamus_stage1.log"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Invoke-LoggedPython {
    param([string[]]$Arguments)
    "[$(Get-Date -Format o)] python $($Arguments -join ' ')" |
        Tee-Object -FilePath $logPath -Append
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $commandExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($commandExitCode -ne 0) {
        throw "Monocular Surface Reconstruction GAMUS Stage-1 command failed with exit code $commandExitCode"
    }
}

Set-Location $projectRoot
$activeTraining = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -match 'train_multidomain\.py'
}
if ($activeTraining) {
    $activeIds = ($activeTraining | Select-Object -ExpandProperty ProcessId) -join ', '
    throw "Another Monocular Surface Reconstruction multidomain trainer is already active (PID: $activeIds)"
}

$latestCheckpoint = Get-ChildItem -Path (Join-Path $projectRoot "experiments") `
    -Directory -Filter "*_multidomain_surface_gamus_stage1_heads" |
    Sort-Object LastWriteTime -Descending |
    ForEach-Object { Join-Path $_.FullName "checkpoint_latest.pt" } |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1

if ($latestCheckpoint) {
    $savedConfig = Join-Path (Split-Path -Parent $latestCheckpoint) "config.yaml"
    Invoke-LoggedPython -Arguments @(
        "-u",
        "scripts\train_multidomain.py",
        "--config", $savedConfig,
        "--resume", $latestCheckpoint
    )
}
else {
    Invoke-LoggedPython -Arguments @(
        "-u",
        "scripts\train_multidomain.py",
        "--config", "configs\multidomain_surface_gamus_stage1_heads.yaml"
    )
}

"[$(Get-Date -Format o)] GAMUS Stage-1 protected-head training complete" |
    Tee-Object -FilePath $logPath -Append

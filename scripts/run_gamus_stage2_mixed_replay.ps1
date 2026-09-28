$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot "configs\multidomain_surface_gamus_stage2_mixed_replay.yaml"
$warmStartPath = Join-Path $projectRoot "experiments\20260912T153331Z_multidomain_surface_gamus_stage1_heads\checkpoint_best_landscape.pt"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "gamus_stage2_mixed_replay.log"
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
        throw "Monocular Surface Reconstruction GAMUS Stage-2 command failed with exit code $commandExitCode"
    }
}

Set-Location $projectRoot
if (-not (Test-Path -LiteralPath $python)) {
    throw "Monocular Surface Reconstruction virtual-environment Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Stage-2 configuration is missing: $configPath"
}
if (-not (Test-Path -LiteralPath $warmStartPath)) {
    throw "Verified Stage-1 warm-start checkpoint is missing: $warmStartPath"
}

$activeTraining = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -match 'train_multidomain\.py'
}
if ($activeTraining) {
    $activeIds = ($activeTraining | Select-Object -ExpandProperty ProcessId) -join ', '
    throw "Another Monocular Surface Reconstruction multidomain trainer is already active (PID: $activeIds)"
}

$latestCheckpoint = Get-ChildItem -Path (Join-Path $projectRoot "experiments") `
    -Directory -Filter "*_multidomain_surface_gamus_stage2_mixed_replay" |
    Sort-Object Name -Descending |
    ForEach-Object { Join-Path $_.FullName "checkpoint_latest.pt" } |
    Where-Object {
        (Test-Path -LiteralPath $_) -and
        (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $_) "config.yaml"))
    } |
    Select-Object -First 1

try {
    if ($latestCheckpoint) {
        $savedConfig = Join-Path (Split-Path -Parent $latestCheckpoint) "config.yaml"
        "[$(Get-Date -Format o)] RESUME $latestCheckpoint" |
            Tee-Object -FilePath $logPath -Append
        Invoke-LoggedPython -Arguments @(
            "-u",
            "scripts\train_multidomain.py",
            "--config", $savedConfig,
            "--resume", $latestCheckpoint
        )
    }
    else {
        "[$(Get-Date -Format o)] START guarded Stage-2 mixed replay" |
            Tee-Object -FilePath $logPath -Append
        Invoke-LoggedPython -Arguments @(
            "-u",
            "scripts\train_multidomain.py",
            "--config", "configs\multidomain_surface_gamus_stage2_mixed_replay.yaml"
        )
    }

    "[$(Get-Date -Format o)] COMPLETE GAMUS Stage-2 guarded mixed-replay training" |
        Tee-Object -FilePath $logPath -Append
}
catch {
    "[$(Get-Date -Format o)] FAILED $($_.Exception.Message)" |
        Tee-Object -FilePath $logPath -Append
    throw
}

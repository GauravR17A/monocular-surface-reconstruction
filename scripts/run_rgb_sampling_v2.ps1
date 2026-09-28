param([string]$ResumePair, [switch]$PreflightOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$logDirectory = Join-Path $projectRoot 'outputs\orchestration'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$lockPath = Join-Path $logDirectory 'rgb_sampling_v2.lock'
$logPath = Join-Path $logDirectory 'rgb_sampling_v2.log'
$runLock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
Push-Location $projectRoot
try {
    $trainers = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'train_[^\s"]*\.py' }
    if ($trainers) { throw "Another trainer is active: $(($trainers.ProcessId) -join ', ')" }
    $arguments = @('-u', (Join-Path $projectRoot 'scripts\train_rgb_sampling_v2.py'))
    if ($PreflightOnly) { $arguments += '--preflight-only' }
    if ($ResumePair) {
        $pairPath = (Resolve-Path -LiteralPath $ResumePair).Path
        $arguments += @('--resume-pair', $pairPath, '--config', (Join-Path $pairPath 'config.yaml'))
    }
    "[$(Get-Date -Format o)] START paired RGB crop experiment; protected app unchanged" | Tee-Object -FilePath $logPath -Append
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & (Join-Path $projectRoot '.venv\Scripts\python.exe') @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $runExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($runExitCode -ne 0) { throw "Paired trainer exited with code $runExitCode. Read status and committed checkpoint before recovery." }
    "[$(Get-Date -Format o)] COMPLETE paired experiment. No app promotion." | Tee-Object -FilePath $logPath -Append
}
finally {
    Pop-Location
    $runLock.Dispose()
}

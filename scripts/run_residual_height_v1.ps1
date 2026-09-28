param(
    [string]$ResumeCheckpoint,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$trainerPath = Join-Path $projectRoot "scripts\train_residual_height.py"
$configPath = Join-Path $projectRoot "configs\residual_height_v1.yaml"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "residual_height_v1.log"
$lockPath = Join-Path $logDirectory "residual_height_v1.lock"
$pointerPath = Join-Path $projectRoot "outputs\runtime\showcase_checkpoint.txt"
$protectedPath = Join-Path $projectRoot "experiments\20260829T203146Z_multidomain_surface_v2_guarded_vegetation\checkpoint_best_guarded_vegetation.pt"
$expectedPointer = "a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9"
$expectedProtected = "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Write-RunLog([string]$Message) {
    "[$(Get-Date -Format o)] $Message" | Tee-Object -FilePath $logPath -Append
}

function Assert-ProductionProtected {
    if ((Get-FileHash -LiteralPath $pointerPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedPointer) {
        throw "Protected application pointer changed. Refusing to continue."
    }
    if ((Get-FileHash -LiteralPath $protectedPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedProtected) {
        throw "Protected production checkpoint changed. Refusing to continue."
    }
}

function Invoke-LoggedPython([string[]]$Arguments) {
    Write-RunLog "python $($Arguments -join ' ')"
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $pythonPath @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $commandExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($commandExitCode -ne 0) { throw "Python exited with code $commandExitCode" }
}

$runLock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
Push-Location $projectRoot
try {
    Assert-ProductionProtected
    $activeTrainers = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        $_.CommandLine -match '(train_residual_height|train_multidomain|train_gamus)[^\s]*\.py'
    }
    if ($activeTrainers) {
        throw "Another height/classification trainer is active: $(($activeTrainers.ProcessId) -join ', ')"
    }
    $trainingArguments = @("-u", $trainerPath, "--config", $configPath, "--device", "cuda")
    if ($ResumeCheckpoint) {
        $checkpointPath = (Resolve-Path -LiteralPath $ResumeCheckpoint).Path
        $experimentRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot "experiments")).Path + [System.IO.Path]::DirectorySeparatorChar
        if (-not $checkpointPath.StartsWith($experimentRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
            [System.IO.Path]::GetFileName($checkpointPath) -ne "checkpoint_latest.pt" -or
            [System.IO.Path]::GetFileName([System.IO.Path]::GetDirectoryName($checkpointPath)) -notlike "*_residual_height_v1") {
            throw "Resume requires this project's residual_height_v1 checkpoint_latest.pt."
        }
        $configPath = Join-Path ([System.IO.Path]::GetDirectoryName($checkpointPath)) "config.yaml"
        $trainingArguments = @("-u", $trainerPath, "--config", $configPath, "--device", "cuda", "--resume", $checkpointPath)
    }
    Write-RunLog "START residual-height controlled experiment; protected model frozen; automatic promotion disabled"
    Invoke-LoggedPython @("-u", $trainerPath, "--config", $configPath, "--preflight-only", "--device", "cpu")
    if (-not $PreflightOnly) {
        # The Python trainer authenticates the passing training-only proof before a fresh run.
        Invoke-LoggedPython $trainingArguments
        Write-RunLog "COMPLETE residual-height run. Review guarded validation; do not promote automatically."
    }
}
catch {
    Write-RunLog "FAILED $($_.Exception.Message)"
    throw
}
finally {
    try {
        Assert-ProductionProtected
        Write-RunLog "VERIFIED production checkpoint and application pointer unchanged"
    }
    finally {
        Pop-Location
        $runLock.Dispose()
    }
}

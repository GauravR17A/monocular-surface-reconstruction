param(
    [switch]$TrainOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $projectRoot "scripts\train_multidomain.py"
$preflight = Join-Path $projectRoot "scripts\preflight_corrected_height_replay_v1.py"
$configPath = Join-Path $projectRoot "configs\multidomain_surface_corrected_height_replay_v1.yaml"
$warmStart = Join-Path $projectRoot "experiments\20260829T203146Z_multidomain_surface_v2_guarded_vegetation\checkpoint_best_guarded_vegetation.pt"
$expectedWarmStartSha256 = "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
$pointerPath = Join-Path $projectRoot "outputs\runtime\showcase_checkpoint.txt"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "corrected_height_replay_v1.log"
$experimentPattern = "*_multidomain_surface_corrected_height_replay_v1"

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Write-ReplayLog {
    param([string]$Message)
    "[$(Get-Date -Format o)] $Message" | Tee-Object -FilePath $logPath -Append
}

function Invoke-LoggedCommand {
    param([string[]]$Arguments)
    Write-ReplayLog "python $($Arguments -join ' ')"
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($exitCode -ne 0) {
        throw "Command failed with exit code $exitCode"
    }
}

function Resolve-ProjectRelativePath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $PathValue))
}

foreach ($requiredFile in @($python, $trainer, $preflight, $configPath, $warmStart, $pointerPath)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Corrected replay is not ready; missing: $requiredFile"
    }
}

$warmStartSha256 = (Get-FileHash -LiteralPath $warmStart -Algorithm SHA256).Hash.ToLowerInvariant()
if ($warmStartSha256 -ne $expectedWarmStartSha256) {
    throw "Protected warm-start checkpoint hash mismatch: $warmStartSha256"
}

$activeTraining = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.ProcessId -ne $PID -and
    $_.CommandLine -match 'train_multidomain\.py'
}
if ($activeTraining) {
    $activeIds = ($activeTraining | Select-Object -ExpandProperty ProcessId) -join ", "
    throw "A Monocular Surface Reconstruction multidomain trainer is already active (PID: $activeIds)"
}

$latestExperiment = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
    -Directory -Filter $experimentPattern -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    Select-Object -First 1
$latestCheckpoint = if ($latestExperiment) {
    Join-Path $latestExperiment.FullName "checkpoint_latest.pt"
}
else {
    $null
}

if ($Resume) {
    if (-not $latestCheckpoint -or -not (Test-Path -LiteralPath $latestCheckpoint -PathType Leaf)) {
        throw "-Resume requested, but no corrected-replay checkpoint exists"
    }
    $selectedConfig = Join-Path $latestExperiment.FullName "config.yaml"
    if (-not (Test-Path -LiteralPath $selectedConfig -PathType Leaf)) {
        throw "Resume checkpoint has no immutable config: $selectedConfig"
    }
}
else {
    if ($latestCheckpoint -and (Test-Path -LiteralPath $latestCheckpoint -PathType Leaf)) {
        throw "A prior corrected replay exists. Use -Resume instead of duplicating it."
    }
    $selectedConfig = $configPath
}

$pointerBefore = Get-Content -LiteralPath $pointerPath -Raw
$pointerFileHashBefore = (Get-FileHash -LiteralPath $pointerPath -Algorithm SHA256).Hash
$liveCheckpointBefore = Resolve-ProjectRelativePath ($pointerBefore.Trim())
$liveCheckpointHashBefore = (Get-FileHash -LiteralPath $liveCheckpointBefore -Algorithm SHA256).Hash
$preflightStamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$preflightReport = Join-Path $logDirectory "corrected_height_replay_v1_preflight_$preflightStamp.json"

Push-Location $projectRoot
try {
    Write-ReplayLog "START corrected GAMUS + measured HighBuild + OpenCanopy replay"
    Write-ReplayLog "Official GAMUS test excluded; promotion disabled"
    Write-ReplayLog "Trainer SHA256: $((Get-FileHash -LiteralPath $trainer -Algorithm SHA256).Hash.ToLowerInvariant())"
    Write-ReplayLog "Config SHA256: $((Get-FileHash -LiteralPath $selectedConfig -Algorithm SHA256).Hash.ToLowerInvariant())"
    if (-not $TrainOnly) {
        Invoke-LoggedCommand -Arguments @(
            "-m", "pytest", "-q",
            "tests\test_surface_dataset.py",
            "tests\test_mixed_replay.py",
            "tests\test_corrected_height_replay_v1_protocol.py",
            "tests\test_domain_surface_model.py"
        )
    }
    Invoke-LoggedCommand -Arguments @(
        "-u", "scripts\preflight_corrected_height_replay_v1.py",
        "--config", $selectedConfig,
        "--output", $preflightReport
    )
    $trainingArguments = @(
        "-u", "scripts\train_multidomain.py",
        "--config", $selectedConfig
    )
    if ($Resume) {
        $trainingArguments += @("--resume", $latestCheckpoint)
        Write-ReplayLog "RESUME $latestCheckpoint"
    }
    Invoke-LoggedCommand -Arguments $trainingArguments
    Write-ReplayLog "COMPLETE corrected height replay; still not promoted"
}
catch {
    Write-ReplayLog "FAILED $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
    $pointerAfter = Get-Content -LiteralPath $pointerPath -Raw
    $pointerFileHashAfter = (Get-FileHash -LiteralPath $pointerPath -Algorithm SHA256).Hash
    $liveCheckpointAfter = Resolve-ProjectRelativePath ($pointerAfter.Trim())
    $liveCheckpointHashAfter = (Get-FileHash -LiteralPath $liveCheckpointAfter -Algorithm SHA256).Hash
    if (
        $pointerBefore -ne $pointerAfter -or
        $pointerFileHashBefore -ne $pointerFileHashAfter -or
        $liveCheckpointBefore -ne $liveCheckpointAfter -or
        $liveCheckpointHashBefore -ne $liveCheckpointHashAfter
    ) {
        Write-ReplayLog "SAFETY FAILURE: live pointer or protected checkpoint changed"
        throw "Live application checkpoint protection failed"
    }
    Write-ReplayLog "VERIFIED live pointer and protected checkpoint are byte-identical"
}

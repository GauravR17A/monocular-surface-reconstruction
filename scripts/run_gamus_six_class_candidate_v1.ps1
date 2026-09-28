param(
    [switch]$TrainOnly,
    [switch]$Resume,
    [string]$HeightPilotExperiment
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $projectRoot "scripts\train_multidomain.py"
$comparator = Join-Path $projectRoot "scripts\compare_gamus_six_class_candidate.py"
$configPath = Join-Path $projectRoot "configs\multidomain_surface_gamus_six_class_candidate_v1.yaml"
$protocolPath = Join-Path $projectRoot "configs\gamus_six_class_comparison_v1.yaml"
$expectedConfigSha256 = "6213accf5334ec1d7cfb394fc6322810242af0d5bab3dc7be3ebba8584b8d01a"
$pointerPath = Join-Path $projectRoot "outputs\runtime\showcase_checkpoint.txt"
$protectedCheckpoint = Join-Path $projectRoot "experiments\20260829T203146Z_multidomain_surface_v2_guarded_vegetation\checkpoint_best_guarded_vegetation.pt"
$expectedProtectedSha256 = "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
$heightLogPath = Join-Path $projectRoot "outputs\orchestration\gamus_direct_height_pilot.log"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "gamus_six_class_candidate_v1.log"
$heightPattern = "*_multidomain_surface_gamus_direct_height_pilot"
$candidatePattern = "*_multidomain_surface_gamus_six_class_candidate_v1"
$comparisonOutput = Join-Path $projectRoot "outputs\evaluation\gamus_six_class_comparison_v1"
$heightLockPath = Join-Path $comparisonOutput "height_pilot_comparison_lock.json"
$launchConfigPath = Join-Path $comparisonOutput "six_class_launch_config.yaml"

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Write-CandidateLog {
    param([string]$Message)
    "[$(Get-Date -Format o)] $Message" | Tee-Object -FilePath $logPath -Append
}

function Invoke-LoggedCommand {
    param([string[]]$Arguments)
    Write-CandidateLog "python $($Arguments -join ' ')"
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

foreach ($requiredFile in @(
    $python,
    $trainer,
    $comparator,
    $configPath,
    $protocolPath,
    $pointerPath,
    $protectedCheckpoint,
    $heightLogPath
)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Six-class experiment is not ready; required file is missing: $requiredFile"
    }
}

$configSha256 = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($configSha256 -ne $expectedConfigSha256) {
    throw "Frozen six-class candidate config hash mismatch: $configSha256"
}
$protectedSha256 = (Get-FileHash -LiteralPath $protectedCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($protectedSha256 -ne $expectedProtectedSha256) {
    throw "Protected checkpoint hash mismatch: $protectedSha256"
}

if ($HeightPilotExperiment) {
    $heightExperiment = Get-Item -LiteralPath (Resolve-ProjectRelativePath $HeightPilotExperiment) -ErrorAction Stop
}
else {
    $heightExperiment = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
        -Directory -Filter $heightPattern -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1
}
if (-not $heightExperiment) {
    throw "The frozen height-focused pilot must finish before the six-class ablation starts"
}

# The direct-height wrapper writes COMPLETE only after training/early-stopping
# exits successfully and its pointer-integrity check passes. Inspect only the
# newest START/RESUME session so an older completion cannot authorize a retry.
$heightLog = @(Get-Content -LiteralPath $heightLogPath -ErrorAction Stop)
$lastHeightStart = -1
for ($index = $heightLog.Count - 1; $index -ge 0; $index--) {
    if ($heightLog[$index] -match "\] (START|RESUME)\b") {
        $lastHeightStart = $index
        break
    }
}
if ($lastHeightStart -lt 0) {
    throw "Height-focused pilot log has no authenticated START/RESUME session"
}
$heightSession = @($heightLog[$lastHeightStart..($heightLog.Count - 1)])
if (-not ($heightSession -match "COMPLETE offline GAMUS direct-height pilot")) {
    throw "The newest height-focused pilot session is not complete"
}
if ($heightSession -match "FAILED ") {
    throw "The newest height-focused pilot session contains a failure"
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

Push-Location $projectRoot
try {
    Invoke-LoggedCommand -Arguments @(
        "-u", "scripts\compare_gamus_six_class_candidate.py",
        "--protocol", $protocolPath,
        "--height-experiment", $heightExperiment.FullName,
        "--preflight-only"
    )
    Invoke-LoggedCommand -Arguments @(
        "-u", "scripts\compare_gamus_six_class_candidate.py",
        "--protocol", $protocolPath,
        "--height-experiment", $heightExperiment.FullName,
        "--seal-height-pilot",
        "--height-lock-output", $heightLockPath,
        "--launch-config-output", $launchConfigPath
    )
}
finally {
    Pop-Location
}

$latestExperiment = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
    -Directory -Filter $candidatePattern -ErrorAction SilentlyContinue |
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
        throw "-Resume was requested, but no resumable six-class checkpoint exists"
    }
    $selectedConfig = Join-Path $latestExperiment.FullName "config.yaml"
    if (-not (Test-Path -LiteralPath $selectedConfig -PathType Leaf)) {
        throw "Resume checkpoint has no immutable saved config: $selectedConfig"
    }
}
else {
    if ($latestCheckpoint -and (Test-Path -LiteralPath $latestCheckpoint -PathType Leaf)) {
        throw "A prior six-class run exists. Use -Resume instead of starting a duplicate."
    }
    if (-not (Test-Path -LiteralPath $launchConfigPath -PathType Leaf)) {
        throw "Height-pilot sealing did not create the immutable launch config"
    }
    $selectedConfig = $launchConfigPath
}

$pointerBefore = Get-Content -LiteralPath $pointerPath -Raw
$pointerFileHashBefore = (Get-FileHash -LiteralPath $pointerPath -Algorithm SHA256).Hash
$liveCheckpointBefore = Resolve-ProjectRelativePath ($pointerBefore.Trim())
if (-not (Test-Path -LiteralPath $liveCheckpointBefore -PathType Leaf)) {
    throw "The live checkpoint pointer target is missing: $liveCheckpointBefore"
}
$liveCheckpointHashBefore = (Get-FileHash -LiteralPath $liveCheckpointBefore -Algorithm SHA256).Hash

Push-Location $projectRoot
try {
    Write-CandidateLog "START isolated GAMUS six-class candidate; official test is not constructed"
    Write-CandidateLog "Height comparison: $($heightExperiment.FullName)"
    Write-CandidateLog "Candidate config SHA256: $configSha256; protected checkpoint SHA256: $protectedSha256"
    Write-CandidateLog "Live pointer will remain untouched; auto-promotion is forbidden"
    if (-not $TrainOnly) {
        Invoke-LoggedCommand -Arguments @(
            "-m", "pytest", "-q",
            "tests\test_gamus_dataset.py",
            "tests\test_train_multidomain_gamus.py",
            "tests\test_domain_surface_model.py",
            "tests\test_classification_metrics.py",
            "tests\test_compare_gamus_six_class_candidate.py"
        )
    }

    $trainingArguments = @(
        "-u", "scripts\train_multidomain.py",
        "--config", $selectedConfig
    )
    if ($Resume) {
        $trainingArguments += @("--resume", $latestCheckpoint)
        Write-CandidateLog "RESUME $latestCheckpoint"
    }
    Invoke-LoggedCommand -Arguments $trainingArguments

    $completedExperiment = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
        -Directory -Filter $candidatePattern -ErrorAction Stop |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if (-not $completedExperiment) {
        throw "Trainer exited without creating a six-class experiment directory"
    }
    Invoke-LoggedCommand -Arguments @(
        "-u", "scripts\compare_gamus_six_class_candidate.py",
        "--protocol", $protocolPath,
        "--height-experiment", $heightExperiment.FullName,
        "--expanded-experiment", $completedExperiment.FullName
    )
    Write-CandidateLog "COMPLETE offline GAMUS six-class candidate and GAMUS comparison"
    Write-CandidateLog "Corrected legacy and unseen-geography evidence remain mandatory external steps"
}
catch {
    Write-CandidateLog "FAILED $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
    $pointerAfter = Get-Content -LiteralPath $pointerPath -Raw
    $pointerFileHashAfter = (Get-FileHash -LiteralPath $pointerPath -Algorithm SHA256).Hash
    $liveCheckpointAfter = Resolve-ProjectRelativePath ($pointerAfter.Trim())
    $liveCheckpointHashAfter = if (Test-Path -LiteralPath $liveCheckpointAfter -PathType Leaf) {
        (Get-FileHash -LiteralPath $liveCheckpointAfter -Algorithm SHA256).Hash
    }
    else {
        $null
    }
    if (
        $pointerBefore -ne $pointerAfter -or
        $pointerFileHashBefore -ne $pointerFileHashAfter -or
        $liveCheckpointBefore -ne $liveCheckpointAfter -or
        $liveCheckpointHashBefore -ne $liveCheckpointHashAfter
    ) {
        Write-CandidateLog "SAFETY FAILURE: live pointer or protected checkpoint changed"
        throw "Live application checkpoint protection failed"
    }
    Write-CandidateLog "VERIFIED live pointer and protected checkpoint are byte-identical"
}

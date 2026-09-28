param(
    [switch]$TrainOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $projectRoot "scripts\train_multidomain.py"
$configPath = Join-Path $projectRoot "configs\multidomain_surface_gamus_direct_height_pilot.yaml"
$approvedIndex = "D:\MSRData\GAMUS_quality_contract_v1_1\approved_samples.json"
$expectedApprovedIndexSha256 = "5320d2e97be357b1e1725d7f2d9640493522ae3d13f1a051b63de55b530c05aa"
$warmStart = Join-Path $projectRoot "experiments\20260829T203146Z_multidomain_surface_v2_guarded_vegetation\checkpoint_best_guarded_vegetation.pt"
$expectedWarmStartSha256 = "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
$pointerPath = Join-Path $projectRoot "outputs\runtime\showcase_checkpoint.txt"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "gamus_direct_height_pilot.log"
$experimentPattern = "*_multidomain_surface_gamus_direct_height_pilot"

New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Write-PilotLog {
    param([string]$Message)
    "[$(Get-Date -Format o)] $Message" | Tee-Object -FilePath $logPath -Append
}

function Invoke-LoggedCommand {
    param([string[]]$Arguments)
    Write-PilotLog "python $($Arguments -join ' ')"
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

foreach ($requiredFile in @($python, $trainer, $configPath, $approvedIndex, $warmStart, $pointerPath)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "GAMUS direct-height pilot is not ready; required file is missing: $requiredFile"
    }
}

$warmStartSha256 = (Get-FileHash -LiteralPath $warmStart -Algorithm SHA256).Hash.ToLowerInvariant()
if ($warmStartSha256 -ne $expectedWarmStartSha256) {
    throw "Protected warm-start checkpoint hash mismatch: $warmStartSha256"
}
$approvedIndexSha256 = (Get-FileHash -LiteralPath $approvedIndex -Algorithm SHA256).Hash.ToLowerInvariant()
if ($approvedIndexSha256 -ne $expectedApprovedIndexSha256) {
    throw "Approved GAMUS quality-contract hash mismatch: $approvedIndexSha256"
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
        throw "-Resume was requested, but no resumable direct-height checkpoint exists"
    }
    $selectedConfig = Join-Path $latestExperiment.FullName "config.yaml"
    if (-not (Test-Path -LiteralPath $selectedConfig -PathType Leaf)) {
        throw "Resume checkpoint has no immutable saved config: $selectedConfig"
    }
}
else {
    if ($latestCheckpoint -and (Test-Path -LiteralPath $latestCheckpoint -PathType Leaf)) {
        throw "A prior direct-height run exists. Use -Resume instead of starting a duplicate."
    }
    $selectedConfig = $configPath
}

$pointerBefore = Get-Content -LiteralPath $pointerPath -Raw
$pointerFileHashBefore = (Get-FileHash -LiteralPath $pointerPath -Algorithm SHA256).Hash
$liveCheckpointBefore = Resolve-ProjectRelativePath ($pointerBefore.Trim())
$liveCheckpointHashBefore = if (Test-Path -LiteralPath $liveCheckpointBefore -PathType Leaf) {
    (Get-FileHash -LiteralPath $liveCheckpointBefore -Algorithm SHA256).Hash
}
else {
    $null
}

Push-Location $projectRoot
try {
    Write-PilotLog "START isolated GAMUS direct-height pilot; official test is not configured"
    Write-PilotLog "Config SHA256: $((Get-FileHash -LiteralPath $selectedConfig -Algorithm SHA256).Hash.ToLowerInvariant())"
    Write-PilotLog "Warm-start SHA256: $warmStartSha256; approved-index SHA256: $approvedIndexSha256"
    Write-PilotLog "Live pointer will remain untouched"
    if (-not $TrainOnly) {
        Invoke-LoggedCommand -Arguments @(
            "-m", "pytest", "-q",
            "tests\test_gamus_dataset.py",
            "tests\test_train_multidomain_gamus.py",
            "tests\test_domain_surface_model.py"
        )
    }

    $trainingArguments = @(
        "-u", "scripts\train_multidomain.py",
        "--config", $selectedConfig
    )
    if ($Resume) {
        $trainingArguments += @("--resume", $latestCheckpoint)
        Write-PilotLog "RESUME $latestCheckpoint"
    }
    Invoke-LoggedCommand -Arguments $trainingArguments
    Write-PilotLog "COMPLETE offline GAMUS direct-height pilot"
}
catch {
    Write-PilotLog "FAILED $($_.Exception.Message)"
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
        Write-PilotLog "SAFETY FAILURE: live pointer or protected checkpoint changed"
        throw "Live application checkpoint protection failed"
    }
    Write-PilotLog "VERIFIED live pointer and protected checkpoint are byte-identical"
}

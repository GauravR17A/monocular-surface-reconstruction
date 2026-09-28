param(
    [switch]$TrainOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot "configs\multidomain_surface_gamus_six_class_head_only_geographic_v1.yaml"
$sourceConfig = Join-Path $projectRoot "configs\multidomain_surface_gamus_six_class_head_only_v2.yaml"
$evaluationProtocol = Join-Path $projectRoot "configs\gamus_locked_nyc_evaluation_v1.yaml"
$trainer = Join-Path $projectRoot "scripts\train_multidomain.py"
$preflight = Join-Path $projectRoot "scripts\preflight_gamus_geographic_candidate.py"
$auditor = Join-Path $projectRoot "scripts\audit_gamus_head_only_checkpoint.py"
$lockedEvaluator = Join-Path $projectRoot "scripts\evaluate_gamus_locked_nyc.py"
$pointerPath = Join-Path $projectRoot "outputs\runtime\showcase_checkpoint.txt"
$protectedCheckpoint = Join-Path $projectRoot "experiments\20260829T203146Z_multidomain_surface_v2_guarded_vegetation\checkpoint_best_guarded_vegetation.pt"
$learningIndex = Join-Path $projectRoot "outputs\data_audits\gamus_geographic_holdout_city_nyc_v1\learning_approved_samples.json"
$holdoutContract = Join-Path $projectRoot "outputs\data_audits\gamus_geographic_holdout_city_nyc_v1\contract.json"
$learningSampling = Join-Path $projectRoot "outputs\data_audits\gamus_head_only_geographic_candidate_v1\gamus_learning_six_class_sampling_index_v1.jsonl"
$holdoutIndex = Join-Path $projectRoot "outputs\data_audits\gamus_head_only_geographic_candidate_v1\nyc_holdout_approved_samples.json"
$comparisonReport = Join-Path $projectRoot "outputs\evaluation\gamus_six_class_comparison_v1\report.json"
$preflightReport = Join-Path $projectRoot "outputs\data_audits\gamus_head_only_geographic_candidate_v1\candidate_preflight.json"
$logPath = Join-Path $projectRoot "outputs\orchestration\gamus_head_only_geographic_v1.log"
$auditRoot = Join-Path $projectRoot "outputs\evaluation\gamus_head_only_geographic_v1"
$lockedRoot = Join-Path $projectRoot "outputs\evaluation\gamus_locked_nyc_v1"
$experimentPattern = "*_multidomain_surface_gamus_six_class_head_only_geographic_v1"

$expectedConfigSha256 = "e14a7cd1189d697ea30c3c437792b73e7287ce2c3acda24219a418d9f9df46ca"
$expectedSourceConfigSha256 = "97ed2e6c97e4d00b8a6929a06bf1c0821de314d32d3a0f2afa85b119cb99c2ca"
$expectedEvaluationProtocolSha256 = "ea24f5f2677c1f82239112ff01254dde6322a4d5f5272e8f9100088bd8029104"
$expectedLockedEvaluatorSha256 = "ef304591d96b93f67abcab5ec841f080b8ee36d8aeb275e8ea847d470c5d5d52"
$expectedProtectedSha256 = "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
$expectedPointerSha256 = "a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9"
$expectedLearningIndexSha256 = "d56764bc9a3cde31760189b1da40cf207de1b9142c7531b24190ed6e2312e5d9"
$expectedHoldoutContractSha256 = "a8a5b0dfaa3a6338a590a12dc7eeab74734f26b465e526b05a15833e9cf02435"
$expectedLearningSamplingSha256 = "32a71cc1cb2d9da75ce8774db5e4db342512991ebb37e4b461f776fe7196eedf"
$expectedHoldoutIndexSha256 = "98281b990b5ccddfda4349d0b43f0216455a8bebca085bf1232abab10f689417"
$expectedComparisonSha256 = "c487424ddd1bb5032a8a0d3e719254b819623cd64d7940333c53cb9269127e09"

New-Item -ItemType Directory -Force -Path (Split-Path $logPath) | Out-Null
New-Item -ItemType Directory -Force -Path $auditRoot | Out-Null

function Write-RunLog {
    param([string]$Message)
    "[$(Get-Date -Format o)] $Message" | Tee-Object -FilePath $logPath -Append
}

function Assert-Hash {
    param([string]$Path, [string]$Expected, [string]$Role)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Role is missing: $Path"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "$Role hash mismatch: expected $Expected, found $actual"
    }
}

function Resolve-ProjectPath {
    param([string]$Value)
    $clean = $Value.TrimStart([char]0xFEFF).Trim()
    if ([System.IO.Path]::IsPathRooted($clean)) {
        return [System.IO.Path]::GetFullPath($clean)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $clean))
}

function Invoke-LoggedPython {
    param([string[]]$Arguments)
    Write-RunLog "python $($Arguments -join ' ')"
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    if ($exitCode -ne 0) {
        throw "Python command failed with exit code $exitCode"
    }
}

Assert-Hash $configPath $expectedConfigSha256 "NYC-blind candidate config"
Assert-Hash $sourceConfig $expectedSourceConfigSha256 "frozen source recipe"
Assert-Hash $evaluationProtocol $expectedEvaluationProtocolSha256 "locked NYC protocol"
Assert-Hash $lockedEvaluator $expectedLockedEvaluatorSha256 "locked NYC evaluator"
Assert-Hash $protectedCheckpoint $expectedProtectedSha256 "protected checkpoint"
Assert-Hash $pointerPath $expectedPointerSha256 "live pointer"
Assert-Hash $learningIndex $expectedLearningIndexSha256 "DC+PHL approved index"
Assert-Hash $holdoutContract $expectedHoldoutContractSha256 "geographic contract"
Assert-Hash $learningSampling $expectedLearningSamplingSha256 "DC+PHL water-sampling index"
Assert-Hash $holdoutIndex $expectedHoldoutIndexSha256 "locked NYC approved index"
Assert-Hash $comparisonReport $expectedComparisonSha256 "v1 comparison report"

$pointerBefore = Get-Content -LiteralPath $pointerPath -Raw
$pointerTargetBefore = Resolve-ProjectPath $pointerBefore
if ($pointerTargetBefore -ne [System.IO.Path]::GetFullPath($protectedCheckpoint)) {
    throw "The live pointer no longer targets the protected checkpoint"
}
$pointerHashBefore = (Get-FileHash -LiteralPath $pointerPath -Algorithm SHA256).Hash
$protectedHashBefore = (Get-FileHash -LiteralPath $protectedCheckpoint -Algorithm SHA256).Hash

$activeTrainer = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.ProcessId -ne $PID -and
    $_.CommandLine -match 'train_multidomain\.py'
}
if ($activeTrainer) {
    throw "Another Monocular Surface Reconstruction trainer is active; the geographic candidate will not start"
}

$existing = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
    -Directory -Filter $experimentPattern -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    Select-Object -First 1
if ($Resume) {
    if (-not $existing -or -not (Test-Path -LiteralPath (Join-Path $existing.FullName "checkpoint_latest.pt"))) {
        throw "-Resume requires an existing geographic checkpoint_latest.pt"
    }
}
elseif ($existing) {
    throw "A geographic experiment already exists; use -Resume rather than duplicating it"
}

Push-Location $projectRoot
try {
    Write-RunLog "START NYC-blind DC+PHL head-only candidate"
    Write-RunLog "NYC is locked; official GAMUS test is forbidden; app promotion is forbidden"
    Invoke-LoggedPython -Arguments @(
        "-u", $preflight,
        "--config", $configPath,
        "--output", $preflightReport
    )
    if (-not $TrainOnly) {
        Invoke-LoggedPython -Arguments @(
            "-m", "pytest", "-q",
            "tests\test_gamus_geographic_holdout.py",
            "tests\test_gamus_geographic_candidate.py",
            "tests\test_gamus_locked_nyc_protocol.py",
            "tests\test_gamus_dataset.py",
            "tests\test_train_multidomain_gamus.py",
            "tests\test_audit_gamus_head_only_checkpoint.py"
        )
    }

    $trainingArguments = @("-u", $trainer, "--config", $configPath)
    if ($Resume) {
        $trainingArguments += @("--resume", (Join-Path $existing.FullName "checkpoint_latest.pt"))
        Write-RunLog "RESUME $($existing.FullName)"
    }
    Invoke-LoggedPython -Arguments $trainingArguments

    $completed = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
        -Directory -Filter $experimentPattern -ErrorAction Stop |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if (-not $completed) {
        throw "Trainer exited without a geographic experiment directory"
    }
    $latest = Join-Path $completed.FullName "checkpoint_latest.pt"
    $best = Join-Path $completed.FullName "checkpoint_best_landscape.pt"
    foreach ($candidate in @($latest, $best)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "Required candidate checkpoint is missing: $candidate"
        }
        $label = [System.IO.Path]::GetFileNameWithoutExtension($candidate)
        Invoke-LoggedPython -Arguments @(
            "-u", $auditor,
            "--base-checkpoint", $protectedCheckpoint,
            "--candidate-checkpoint", $candidate,
            "--comparison-report", $comparisonReport,
            "--output", (Join-Path $auditRoot "$($label)_audit.json")
        )
    }

    # Freeze validation-selected weights and all predeclared thresholds, but do
    # not consume the NYC holdout. That remains a separate explicit one-shot.
    Invoke-LoggedPython -Arguments @(
        "-u", $lockedEvaluator,
        "--freeze-only",
        "--protocol", $evaluationProtocol,
        "--candidate-checkpoint", $best,
        "--output-root", $lockedRoot
    )
    Write-RunLog "COMPLETE candidate; NYC freeze created but holdout not consumed"
}
catch {
    Write-RunLog "FAILED $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
    $pointerAfter = Get-Content -LiteralPath $pointerPath -Raw
    $pointerHashAfter = (Get-FileHash -LiteralPath $pointerPath -Algorithm SHA256).Hash
    $protectedHashAfter = (Get-FileHash -LiteralPath $protectedCheckpoint -Algorithm SHA256).Hash
    if (
        $pointerBefore -ne $pointerAfter -or
        $pointerHashBefore -ne $pointerHashAfter -or
        $protectedHashBefore -ne $protectedHashAfter
    ) {
        Write-RunLog "SAFETY FAILURE: live pointer or protected checkpoint changed"
        throw "Live application model protection failed"
    }
    Write-RunLog "VERIFIED live pointer and protected checkpoint are unchanged"
}

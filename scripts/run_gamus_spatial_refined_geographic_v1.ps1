param(
    [switch]$TrainOnly,
    [switch]$Resume,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$config = Join-Path $projectRoot "configs\multidomain_surface_gamus_six_class_spatial_refined_geographic_v1.yaml"
$sourceConfig = Join-Path $projectRoot "configs\multidomain_surface_gamus_six_class_spatial_refined_v3.yaml"
$evaluationProtocol = Join-Path $projectRoot "configs\gamus_locked_nyc_spatial_refined_evaluation_v1.yaml"
$trainer = Join-Path $projectRoot "scripts\train_multidomain.py"
$modelCode = Join-Path $projectRoot "src\msr\models\domain_surface_net.py"
$lossCode = Join-Path $projectRoot "src\msr\training\losses.py"
$preflight = Join-Path $projectRoot "scripts\preflight_gamus_geographic_candidate.py"
$geographicModule = Join-Path $projectRoot "src\msr\data\gamus_geographic_candidate.py"
$auditor = Join-Path $projectRoot "scripts\audit_gamus_spatial_refined_head_checkpoint.py"
$lockedEvaluator = Join-Path $projectRoot "scripts\evaluate_gamus_locked_nyc.py"
$pointer = Join-Path $projectRoot "outputs\runtime\showcase_checkpoint.txt"
$protected = Join-Path $projectRoot "experiments\20260829T203146Z_multidomain_surface_v2_guarded_vegetation\checkpoint_best_guarded_vegetation.pt"
$learningIndex = Join-Path $projectRoot "outputs\data_audits\gamus_geographic_holdout_city_nyc_v1\learning_approved_samples.json"
$holdoutContract = Join-Path $projectRoot "outputs\data_audits\gamus_geographic_holdout_city_nyc_v1\contract.json"
$learningSampling = Join-Path $projectRoot "outputs\data_audits\gamus_head_only_geographic_candidate_v1\gamus_learning_six_class_sampling_index_v1.jsonl"
$holdoutIndex = Join-Path $projectRoot "outputs\data_audits\gamus_head_only_geographic_candidate_v1\nyc_holdout_approved_samples.json"
$comparison = Join-Path $projectRoot "outputs\evaluation\gamus_six_class_comparison_v1\report.json"
$preflightReport = Join-Path $projectRoot "outputs\data_audits\gamus_spatial_refined_geographic_candidate_v1\candidate_preflight.json"
$auditRoot = Join-Path $projectRoot "outputs\evaluation\gamus_spatial_refined_geographic_v1"
$lockedRoot = Join-Path $projectRoot "outputs\evaluation\gamus_locked_nyc_spatial_refined_v1"
$log = Join-Path $projectRoot "outputs\orchestration\gamus_spatial_refined_geographic_v1.log"
$experimentPattern = "*_multidomain_surface_gamus_six_class_spatial_refined_geographic_v1"

$expectedConfigSha256 = "265f22f6d457ce2b3335061a739a95304cbf176bd9fe2d0cfc413d009c8d9f63"
$expectedSourceConfigSha256 = "a91acb93ef8a9e6c376ba1b99b71e0a8390ab6132aeda71790e433e9bdf55946"
$expectedEvaluationProtocolSha256 = "0af0313686fe62b3d09ae5ea132a4a448e453306f027b66003b724a0aacfc503"
$expectedTrainerSha256 = "cce6e684f265f491a238564b59c9220e328b873e4a151fcfdf622c1c3de3ce7e"
$expectedPreflightSha256 = "e276dfc43f1d12ab80ea0790633fb2376e86c6388cc2130cbb3406f588755c34"
$expectedGeographicModuleSha256 = "f422167bb30f39ce1ec106efaf6cc948b3e11aa274d5e682813f1e7eb40e0afe"
$expectedModelCodeSha256 = "5958b161b8082653dd708cffa3da1d5ce5aca16d9129f20b8adfab5b96b0c040"
$expectedLossCodeSha256 = "1066e9f05fedf5a364aff09be31eb2d131a988765dccc96a2df321f7207d33ee"
$expectedAuditorSha256 = "d6fa85b87fce3bd7da5221380e2c70f968e63da31716d9536861d759ed88df1f"
$expectedLockedEvaluatorSha256 = "ef304591d96b93f67abcab5ec841f080b8ee36d8aeb275e8ea847d470c5d5d52"
$expectedProtectedSha256 = "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
$expectedPointerSha256 = "a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9"
$expectedLearningIndexSha256 = "d56764bc9a3cde31760189b1da40cf207de1b9142c7531b24190ed6e2312e5d9"
$expectedHoldoutContractSha256 = "a8a5b0dfaa3a6338a590a12dc7eeab74734f26b465e526b05a15833e9cf02435"
$expectedLearningSamplingSha256 = "32a71cc1cb2d9da75ce8774db5e4db342512991ebb37e4b461f776fe7196eedf"
$expectedHoldoutIndexSha256 = "98281b990b5ccddfda4349d0b43f0216455a8bebca085bf1232abab10f689417"
$expectedComparisonSha256 = "c487424ddd1bb5032a8a0d3e719254b819623cd64d7940333c53cb9269127e09"

New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
New-Item -ItemType Directory -Force -Path $auditRoot | Out-Null

function Write-RunLog {
    param([string]$Message)
    "[$(Get-Date -Format o)] $Message" | Tee-Object -FilePath $log -Append
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

function Invoke-Python {
    param([string[]]$Arguments)
    Write-RunLog "python $($Arguments -join ' ')"
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    if ($exitCode -ne 0) {
        throw "Python command failed with exit code $exitCode"
    }
}

Assert-Hash $config $expectedConfigSha256 "DC+PHL spatial comparator config"
Assert-Hash $sourceConfig $expectedSourceConfigSha256 "frozen spatial v3 source recipe"
Assert-Hash $evaluationProtocol $expectedEvaluationProtocolSha256 "locked NYC spatial protocol"
Assert-Hash $trainer $expectedTrainerSha256 "trainer"
Assert-Hash $modelCode $expectedModelCodeSha256 "surface-model implementation"
Assert-Hash $lossCode $expectedLossCodeSha256 "loss implementation"
Assert-Hash $preflight $expectedPreflightSha256 "geographic preflight"
Assert-Hash $geographicModule $expectedGeographicModuleSha256 "geographic protocol module"
Assert-Hash $auditor $expectedAuditorSha256 "spatial tensor auditor"
Assert-Hash $lockedEvaluator $expectedLockedEvaluatorSha256 "locked NYC evaluator"
Assert-Hash $protected $expectedProtectedSha256 "protected production checkpoint"
Assert-Hash $pointer $expectedPointerSha256 "live application pointer"
Assert-Hash $learningIndex $expectedLearningIndexSha256 "DC+PHL approved index"
Assert-Hash $holdoutContract $expectedHoldoutContractSha256 "geographic holdout contract"
Assert-Hash $learningSampling $expectedLearningSamplingSha256 "DC+PHL sampling index"
Assert-Hash $holdoutIndex $expectedHoldoutIndexSha256 "locked NYC approved index"
Assert-Hash $comparison $expectedComparisonSha256 "v1 comparison report"

$pointerBefore = Get-Content -LiteralPath $pointer -Raw
$pointerTarget = Resolve-ProjectPath $pointerBefore
if ($pointerTarget -ne [System.IO.Path]::GetFullPath($protected)) {
    throw "The live pointer no longer targets the protected production checkpoint"
}
$pointerHashBefore = (Get-FileHash -LiteralPath $pointer -Algorithm SHA256).Hash
$protectedHashBefore = (Get-FileHash -LiteralPath $protected -Algorithm SHA256).Hash

$activeTrainer = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.ProcessId -ne $PID -and
    $_.CommandLine -match 'train_multidomain\.py'
}
if ($activeTrainer -and -not $PreflightOnly) {
    throw "Another Monocular Surface Reconstruction trainer is active; geographic training will not start"
}

$existing = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
    -Directory -Filter $experimentPattern -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    Select-Object -First 1
if ($Resume) {
    if (-not $existing -or -not (Test-Path -LiteralPath (Join-Path $existing.FullName "checkpoint_latest.pt"))) {
        throw "-Resume requires an existing spatial geographic checkpoint_latest.pt"
    }
}
elseif ($existing -and -not $PreflightOnly) {
    throw "A spatial geographic experiment already exists; use -Resume"
}

Push-Location $projectRoot
try {
    Write-RunLog "PREFLIGHT DC+PHL spatial-refined classifier comparator"
    Write-RunLog "NYC is excluded from this head's training but is not system-unseen; official GAMUS test and app promotion are forbidden"
    Invoke-Python -Arguments @(
        "-u", $preflight,
        "--config", $config,
        "--output", $preflightReport
    )
    if (-not $TrainOnly) {
        Invoke-Python -Arguments @(
            "-m", "pytest", "-q",
            "tests\test_gamus_geographic_holdout.py",
            "tests\test_gamus_geographic_candidate.py",
            "tests\test_gamus_locked_nyc_protocol.py",
            "tests\test_gamus_spatial_refined_v3_protocol.py",
            "tests\test_gamus_dataset.py",
            "tests\test_train_multidomain_gamus.py"
        )
    }
    if ($PreflightOnly) {
        Write-RunLog "PREFLIGHT COMPLETE; no training or holdout access occurred"
        return
    }

    $trainingArguments = @("-u", $trainer, "--config", $config)
    if ($Resume) {
        $trainingArguments += @("--resume", (Join-Path $existing.FullName "checkpoint_latest.pt"))
        Write-RunLog "RESUME $($existing.FullName)"
    }
    Write-RunLog "START DC+PHL-only spatial-refined training"
    Invoke-Python -Arguments $trainingArguments

    $completed = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
        -Directory -Filter $experimentPattern -ErrorAction Stop |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if (-not $completed) {
        throw "Trainer exited without a spatial geographic experiment directory"
    }
    $latest = Join-Path $completed.FullName "checkpoint_latest.pt"
    $best = Join-Path $completed.FullName "checkpoint_best_landscape.pt"
    foreach ($candidate in @($latest, $best)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "Required candidate checkpoint is missing: $candidate"
        }
        $label = [System.IO.Path]::GetFileNameWithoutExtension($candidate)
        Invoke-Python -Arguments @(
            "-u", $auditor,
            "--base-checkpoint", $protected,
            "--candidate-checkpoint", $candidate,
            "--source-recipe-config", $sourceConfig,
            "--comparison-report", $comparison,
            "--output", (Join-Path $auditRoot "$($label)_audit.json")
        )
    }

    Invoke-Python -Arguments @(
        "-u", $lockedEvaluator,
        "--freeze-only",
        "--protocol", $evaluationProtocol,
        "--candidate-checkpoint", $best,
        "--output-root", $lockedRoot
    )
    Write-RunLog "COMPLETE candidate and freeze; NYC holdout remains unconsumed"
}
finally {
    Pop-Location
    $pointerAfter = Get-Content -LiteralPath $pointer -Raw
    $pointerHashAfter = (Get-FileHash -LiteralPath $pointer -Algorithm SHA256).Hash
    $protectedHashAfter = (Get-FileHash -LiteralPath $protected -Algorithm SHA256).Hash
    if (
        $pointerBefore -ne $pointerAfter -or
        $pointerHashBefore -ne $pointerHashAfter -or
        $protectedHashBefore -ne $protectedHashAfter
    ) {
        throw "Protected application state changed during geographic experiment"
    }
    Write-RunLog "VERIFIED live pointer and protected checkpoint are unchanged"
}

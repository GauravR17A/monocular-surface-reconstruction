param(
    [switch]$TrainOnly,
    [switch]$Resume,
    [switch]$PreflightOnly,
    [switch]$SealConfigOnly,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ApprovedBaselineAuditSha256,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ApprovedBaselineReplaySha256,
    [ValidatePattern('^$|^[0-9a-fA-F]{64}$')]
    [string]$ApprovedV4ConfigSha256 = "",
    [ValidatePattern('^$|^[0-9a-fA-F]{64}$')]
    [string]$ApprovedRuntimeDataSealSha256 = ""
)

$ErrorActionPreference = "Stop"
if ($SealConfigOnly -and ($Resume -or $PreflightOnly)) {
    throw "-SealConfigOnly cannot be combined with -Resume or -PreflightOnly"
}
if (-not $SealConfigOnly -and [string]::IsNullOrWhiteSpace($ApprovedV4ConfigSha256)) {
    throw "Training/preflight requires the hash printed by a reviewed -SealConfigOnly run"
}
if (-not $SealConfigOnly -and [string]::IsNullOrWhiteSpace($ApprovedRuntimeDataSealSha256)) {
    throw "Training/preflight requires the externally anchored runtime/data seal hash"
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$config = Join-Path $projectRoot "configs\multidomain_surface_gamus_six_class_hierarchical_v4.yaml"
$comparatorConfig = Join-Path $projectRoot "configs\multidomain_surface_gamus_six_class_spatial_refined_geographic_v1.yaml"
$baselineAudit = Join-Path $projectRoot "outputs\evaluation\gamus_spatial_refined_geographic_v1\checkpoint_best_landscape_audit.json"
$baselineReplay = Join-Path $projectRoot "outputs\evaluation\gamus_dcphl_independent_replay_v1\v3_baseline_report.json"
$comparatorCheckpoint = Join-Path $projectRoot "experiments\20260913T031659Z_multidomain_surface_gamus_six_class_spatial_refined_geographic_v1\checkpoint_best_landscape.pt"
$builder = Join-Path $projectRoot "scripts\build_gamus_hierarchical_v4_config.py"
$preflight = Join-Path $projectRoot "scripts\preflight_gamus_hierarchical_v4.py"
$trainer = Join-Path $projectRoot "scripts\train_multidomain.py"
$modelCode = Join-Path $projectRoot "src\msr\models\domain_surface_net.py"
$lossCode = Join-Path $projectRoot "src\msr\training\losses.py"
$stateAuditor = Join-Path $projectRoot "scripts\audit_gamus_hierarchical_v4_checkpoint.py"
$heightAuditor = Join-Path $projectRoot "scripts\audit_height_output_identity.py"
$pairedReplayEvaluator = Join-Path $projectRoot "scripts\evaluate_gamus_dcphl_paired_replay.py"
$runtimeDataSealer = Join-Path $projectRoot "scripts\seal_gamus_v4_runtime_data.py"
$pointer = Join-Path $projectRoot "outputs\runtime\showcase_checkpoint.txt"
$protected = Join-Path $projectRoot "experiments\20260829T203146Z_multidomain_surface_v2_guarded_vegetation\checkpoint_best_guarded_vegetation.pt"
$preflightReport = Join-Path $projectRoot "outputs\data_audits\gamus_hierarchical_v4\preflight.json"
$runtimeDataSeal = Join-Path $projectRoot "outputs\data_audits\gamus_hierarchical_v4\runtime_data_seal.json"
$auditRoot = Join-Path $projectRoot "outputs\evaluation\gamus_hierarchical_v4"
$freezeRoot = Join-Path $projectRoot "outputs\evaluation\gamus_hierarchical_v4_frozen"
$log = Join-Path $projectRoot "outputs\orchestration\gamus_hierarchical_v4.log"
$experimentPattern = "*_multidomain_surface_gamus_six_class_hierarchical_v4"

# These hashes bind the launcher to the reviewed code.  Any later source edit
# requires a deliberate hash update before training can start.
$expectedComparatorConfigSha256 = "265f22f6d457ce2b3335061a739a95304cbf176bd9fe2d0cfc413d009c8d9f63"
$expectedComparatorCheckpointSha256 = "5f12dc04cc34097dc4770b1821571ccbbae75382198207c07ba027af5b4f3585"
$expectedBuilderSha256 = "75597e305b6bc867687428b2e69ab8d98be05f74f843247c2f85ef0a26e4cff2"
$expectedPreflightSha256 = "226cf2b86d302cde415d1517c78f2cea833caccf964e5209ac1aaf8c9a20e229"
# Reviewed 2026-09-15: opt-in height replay, missing-class evaluation, and feature access preserve the
# default V4 execution path. An old runtime/data seal still needs revalidation.
$expectedTrainerSha256 = "2dc18b785d0c0cfaa2cd1cd89f4c793e65f16328c565545236366732481dc06e"
$expectedModelCodeSha256 = "644b5490b61d2778cc0fbc0b242ba18cf8cf2621b137fecbec1ca0e4dffdb3c1"
$expectedLossCodeSha256 = "1066e9f05fedf5a364aff09be31eb2d131a988765dccc96a2df321f7207d33ee"
$expectedStateAuditorSha256 = "9a0a71057b379fcc52b9489919cf17e3bc7bf0cd3ac0f3cb760de5811316d585"
$expectedHeightAuditorSha256 = "e992894b0eaa63119a2909ac06ae208207aaba892264e88de7647dfed3ed2950"
$expectedPairedReplayEvaluatorSha256 = "452a6a9aeb50c731b195a11ea6bb2af9610c4ef57bc7cf8c76e9f4a8a53c955c"
$expectedRuntimeDataSealerSha256 = "0577844d2261da1ee765102691de54e90178e835a0a34924232da845fef9957d"
$expectedProtectedSha256 = "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
$expectedPointerSha256 = "a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9"

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
    if ($actual -ne $Expected.ToLowerInvariant()) {
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

Assert-Hash $comparatorConfig $expectedComparatorConfigSha256 "paired V3 config"
Assert-Hash $baselineAudit $ApprovedBaselineAuditSha256 "paired V3 baseline audit"
Assert-Hash $baselineReplay $ApprovedBaselineReplaySha256 "paired V3 independent replay"
Assert-Hash $comparatorCheckpoint $expectedComparatorCheckpointSha256 "paired V3 checkpoint"
Assert-Hash $builder $expectedBuilderSha256 "V4 deterministic config builder"
Assert-Hash $preflight $expectedPreflightSha256 "V4 fail-closed preflight"
Assert-Hash $trainer $expectedTrainerSha256 "trainer"
Assert-Hash $modelCode $expectedModelCodeSha256 "model implementation"
Assert-Hash $lossCode $expectedLossCodeSha256 "loss implementation"
Assert-Hash $stateAuditor $expectedStateAuditorSha256 "V4 state/metric auditor"
Assert-Hash $heightAuditor $expectedHeightAuditorSha256 "height-output identity auditor"
Assert-Hash $pairedReplayEvaluator $expectedPairedReplayEvaluatorSha256 "independent paired replay evaluator"
Assert-Hash $runtimeDataSealer $expectedRuntimeDataSealerSha256 "runtime/data provenance sealer"
Assert-Hash $protected $expectedProtectedSha256 "protected production checkpoint"
Assert-Hash $pointer $expectedPointerSha256 "live application pointer"

$pointerBefore = Get-Content -LiteralPath $pointer -Raw
$pointerTarget = Resolve-ProjectPath $pointerBefore
if ($pointerTarget -ne [System.IO.Path]::GetFullPath($protected)) {
    throw "The live pointer no longer targets the protected production checkpoint"
}
$pointerHashBefore = (Get-FileHash -LiteralPath $pointer -Algorithm SHA256).Hash
$protectedHashBefore = (Get-FileHash -LiteralPath $protected -Algorithm SHA256).Hash
$temporaryConfig = Join-Path (Split-Path $config) (
    ".hierarchical_v4.$([guid]::NewGuid().ToString('N')).tmp.yaml"
)

Push-Location $projectRoot
try {
    Write-RunLog "BUILD deterministic V4 config from the authenticated paired V3 audit"
    Invoke-Python -Arguments @(
        "-u", $builder,
        "--source-comparator-config", $comparatorConfig,
        "--fixed-v3-baseline-audit", $baselineAudit,
        "--fixed-v3-independent-replay", $baselineReplay,
        "--output", $temporaryConfig
    )
    $generatedConfigSha256 = (
        Get-FileHash -LiteralPath $temporaryConfig -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if (-not $SealConfigOnly -and $generatedConfigSha256 -ne $ApprovedV4ConfigSha256.ToLowerInvariant()) {
        throw (
            "Generated V4 config hash differs from the reviewed hash: " +
            "$generatedConfigSha256"
        )
    }
    if (Test-Path -LiteralPath $config -PathType Leaf) {
        $existingConfigSha256 = (
            Get-FileHash -LiteralPath $config -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        if ($existingConfigSha256 -ne $generatedConfigSha256) {
            throw "A different sealed V4 config already exists; refusing to overwrite it"
        }
        Remove-Item -LiteralPath $temporaryConfig
    }
    else {
        Move-Item -LiteralPath $temporaryConfig -Destination $config
    }

    Invoke-Python -Arguments @(
        "-u", $preflight,
        "--config", $config,
        "--source-comparator-config", $comparatorConfig,
        "--fixed-v3-baseline-audit", $baselineAudit,
        "--fixed-v3-independent-replay", $baselineReplay,
        "--output", $preflightReport
    )
    if ($SealConfigOnly) {
        Write-RunLog "SEALED CONFIG SHA256 $generatedConfigSha256"
        Write-RunLog "Review this config and rerun with -ApprovedV4ConfigSha256 $generatedConfigSha256"
        return
    }
    Assert-Hash $config $ApprovedV4ConfigSha256 "reviewed V4 config"
    Assert-Hash $runtimeDataSeal $ApprovedRuntimeDataSealSha256 "externally anchored runtime/data seal"
    Invoke-Python -Arguments @(
        "-u", $runtimeDataSealer, "verify",
        "--seal", $runtimeDataSeal,
        "--expected-seal-sha256", $ApprovedRuntimeDataSealSha256,
        "--config", $config,
        "--jobs", "4"
    )

    if (-not $TrainOnly) {
        Invoke-Python -Arguments @(
            "-m", "pytest", "-q",
            "tests\test_gamus_hierarchical_v4.py",
            "tests\test_gamus_hierarchical_v4_protocol.py",
            "tests\test_height_output_identity_audit.py",
            "tests\test_preflight_gamus_hierarchical_v4.py",
            "tests\test_run_gamus_hierarchical_v4.py",
            "tests\test_evaluate_gamus_dcphl_paired_replay.py",
            "tests\test_seal_gamus_v4_runtime_data.py"
        )
    }
    if ($PreflightOnly) {
        Write-RunLog "PREFLIGHT COMPLETE; no training, NYC evaluation, or promotion occurred"
        return
    }

    $activeTrainer = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        $_.ProcessId -ne $PID -and
        $_.CommandLine -match 'train_multidomain\.py'
    }
    if ($activeTrainer) {
        throw "Another Monocular Surface Reconstruction trainer is active; V4 training will not start"
    }

    $existing = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
        -Directory -Filter $experimentPattern -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if ($Resume) {
        if (-not $existing -or -not (Test-Path -LiteralPath (Join-Path $existing.FullName "checkpoint_latest.pt"))) {
            throw "-Resume requires an existing V4 checkpoint_latest.pt"
        }
    }
    elseif ($existing) {
        throw "A V4 experiment already exists; use -Resume"
    }

    $trainingArguments = @("-u", $trainer, "--config", $config)
    if ($Resume) {
        $trainingArguments += @("--resume", (Join-Path $existing.FullName "checkpoint_latest.pt"))
        Write-RunLog "RESUME $($existing.FullName)"
    }
    Write-RunLog "START paired DC+PHL hierarchical-head training; official GAMUS test reuse is forbidden"
    Invoke-Python -Arguments $trainingArguments

    $completed = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
        -Directory -Filter $experimentPattern -ErrorAction Stop |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if (-not $completed) {
        throw "Trainer exited without a V4 experiment directory"
    }
    $latest = Join-Path $completed.FullName "checkpoint_latest.pt"
    $best = Join-Path $completed.FullName "checkpoint_best_landscape.pt"
    $bestAudit = $null
    $bestHeightAudit = $null
    $bestPairedReplay = $null
    foreach ($requiredCandidate in @($latest, $best)) {
        if (-not (Test-Path -LiteralPath $requiredCandidate -PathType Leaf)) {
            throw "Required V4 checkpoint is missing: $requiredCandidate"
        }
    }
    foreach ($candidate in @($latest, $best)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "Required V4 checkpoint is missing: $candidate"
        }
        $label = [System.IO.Path]::GetFileNameWithoutExtension($candidate)
        $candidateSha256 = (
            Get-FileHash -LiteralPath $candidate -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        $heightOutput = Join-Path $auditRoot "$($label)_height_output_identity.json"
        $pairedReplayOutput = Join-Path $auditRoot "$($label)_paired_independent_replay.json"
        $stateOutput = Join-Path $auditRoot "$($label)_audit.json"
        Invoke-Python -Arguments @(
            "-u", $heightAuditor,
            "--protected-checkpoint", $protected,
            "--candidate-checkpoint", $candidate,
            "--expected-protected-sha256", $expectedProtectedSha256,
            "--expected-candidate-sha256", $candidateSha256,
            "--output", $heightOutput
        )
        Invoke-Python -Arguments @(
            "-u", $pairedReplayEvaluator,
            "--v3-checkpoint", $comparatorCheckpoint,
            "--v3-config", $comparatorConfig,
            "--v3-expected-checkpoint-sha256", $expectedComparatorCheckpointSha256,
            "--v3-expected-config-sha256", $expectedComparatorConfigSha256,
            "--v4-checkpoint", $candidate,
            "--v4-config", $config,
            "--v4-expected-checkpoint-sha256", $candidateSha256,
            "--v4-expected-config-sha256", $ApprovedV4ConfigSha256,
            "--device", "cuda",
            "--precision", "fp32",
            "--output", $pairedReplayOutput
        )
        $pairedReplaySha256 = (
            Get-FileHash -LiteralPath $pairedReplayOutput -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        Invoke-Python -Arguments @(
            "-u", $stateAuditor,
            "--base-checkpoint", $protected,
            "--candidate-checkpoint", $candidate,
            "--candidate-config", $config,
            "--expected-candidate-config-sha256", $ApprovedV4ConfigSha256,
            "--source-comparator-config", $comparatorConfig,
            "--fixed-v3-baseline-audit", $baselineAudit,
            "--fixed-v3-independent-replay", $baselineReplay,
            "--paired-independent-replay", $pairedReplayOutput,
            "--expected-paired-independent-replay-sha256", $pairedReplaySha256,
            "--height-output-identity", $heightOutput,
            "--output", $stateOutput
        )
        if ($candidate -eq $best) {
            $bestAudit = $stateOutput
            $bestHeightAudit = $heightOutput
            $bestPairedReplay = $pairedReplayOutput
        }
    }

    Invoke-Python -Arguments @(
        "-u", $runtimeDataSealer, "verify",
        "--seal", $runtimeDataSeal,
        "--expected-seal-sha256", $ApprovedRuntimeDataSealSha256,
        "--config", $config,
        "--jobs", "4"
    )

    $eligibility = Get-Content -LiteralPath $bestAudit -Raw | ConvertFrom-Json
    if ($eligibility.fully_eligible_for_locked_classifier_excluded_city_evaluation -ne $true) {
        throw "V4 failed at least one paired gate; NYC stays closed and no candidate is frozen"
    }
    if ($eligibility.height_output_identity.passes -ne $true) {
        throw "V4 did not prove bit-exact height outputs; NYC stays closed"
    }

    New-Item -ItemType Directory -Force -Path $freezeRoot | Out-Null
    $freezePath = Join-Path $freezeRoot "candidate_freeze_manifest.json"
    if (Test-Path -LiteralPath $freezePath) {
        throw "A V4 candidate freeze already exists; refusing silent replacement"
    }
    $freeze = [ordered]@{
        schema = "msr.gamus_hierarchical_v4_candidate_freeze.v1"
        created_at = (Get-Date).ToUniversalTime().ToString("o")
        candidate_checkpoint = [System.IO.Path]::GetFullPath($best)
        candidate_checkpoint_sha256 = (Get-FileHash -LiteralPath $best -Algorithm SHA256).Hash.ToLowerInvariant()
        config = [System.IO.Path]::GetFullPath($config)
        config_sha256 = $generatedConfigSha256
        paired_v3_baseline_audit = [System.IO.Path]::GetFullPath($baselineAudit)
        paired_v3_baseline_audit_sha256 = $ApprovedBaselineAuditSha256.ToLowerInvariant()
        paired_v3_independent_replay = [System.IO.Path]::GetFullPath($baselineReplay)
        paired_v3_independent_replay_sha256 = $ApprovedBaselineReplaySha256.ToLowerInvariant()
        paired_v3_v4_independent_replay = [System.IO.Path]::GetFullPath($bestPairedReplay)
        paired_v3_v4_independent_replay_sha256 = (Get-FileHash -LiteralPath $bestPairedReplay -Algorithm SHA256).Hash.ToLowerInvariant()
        runtime_data_seal = [System.IO.Path]::GetFullPath($runtimeDataSeal)
        runtime_data_seal_sha256 = $ApprovedRuntimeDataSealSha256.ToLowerInvariant()
        v4_audit = [System.IO.Path]::GetFullPath($bestAudit)
        v4_audit_sha256 = (Get-FileHash -LiteralPath $bestAudit -Algorithm SHA256).Hash.ToLowerInvariant()
        height_output_identity = [System.IO.Path]::GetFullPath($bestHeightAudit)
        height_output_identity_sha256 = (Get-FileHash -LiteralPath $bestHeightAudit -Algorithm SHA256).Hash.ToLowerInvariant()
        paired_gates_passed = $true
        height_outputs_bit_exact = $true
        nyc_interpretation = "excluded_from_this_classifier_training_not_system_unseen"
        nyc_evaluated = $false
        official_test_global_status = "previously_consumed_forbidden_for_reuse"
        official_test_used_by_v4 = $false
        external_system_unseen_geography = "pending_separate_dataset"
        application_model_changed = $false
        promotion_performed = $false
    }
    $freezeTemporary = "$freezePath.tmp"
    [System.IO.File]::WriteAllText(
        $freezeTemporary,
        ($freeze | ConvertTo-Json -Depth 8),
        [System.Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $freezeTemporary -Destination $freezePath
    Write-RunLog "COMPLETE paired V4 candidate frozen; NYC remains unconsumed by this run and the app model is unchanged"
}
finally {
    Pop-Location
    if (Test-Path -LiteralPath $temporaryConfig -PathType Leaf) {
        Remove-Item -LiteralPath $temporaryConfig
    }
    $pointerAfter = Get-Content -LiteralPath $pointer -Raw
    $pointerHashAfter = (Get-FileHash -LiteralPath $pointer -Algorithm SHA256).Hash
    $protectedHashAfter = (Get-FileHash -LiteralPath $protected -Algorithm SHA256).Hash
    if (
        $pointerBefore -ne $pointerAfter -or
        $pointerHashBefore -ne $pointerHashAfter -or
        $protectedHashBefore -ne $protectedHashAfter
    ) {
        throw "Protected application state changed during the V4 experiment"
    }
    Write-RunLog "VERIFIED live pointer and protected checkpoint are unchanged"
}

param(
    [switch]$TrainOnly,
    [switch]$Resume,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot "configs\multidomain_surface_gamus_six_class_spatial_refined_v3.yaml"
$sourceConfigPath = Join-Path $projectRoot "configs\multidomain_surface_gamus_six_class_head_only_v2.yaml"
$trainer = Join-Path $projectRoot "scripts\train_multidomain.py"
$auditor = Join-Path $projectRoot "scripts\audit_gamus_spatial_refined_head_checkpoint.py"
$modelCode = Join-Path $projectRoot "src\msr\models\domain_surface_net.py"
$lossCode = Join-Path $projectRoot "src\msr\training\losses.py"
$datasetCode = Join-Path $projectRoot "src\msr\data\gamus_dataset.py"
$metricsCode = Join-Path $projectRoot "src\msr\evaluation\classification_metrics.py"
$pointerPath = Join-Path $projectRoot "outputs\runtime\showcase_checkpoint.txt"
$protectedCheckpoint = Join-Path $projectRoot "experiments\20260829T203146Z_multidomain_surface_v2_guarded_vegetation\checkpoint_best_guarded_vegetation.pt"
$samplingIndex = Join-Path $projectRoot "outputs\analysis\gamus_train_six_class_tile_index_v1.jsonl"
$comparisonReport = Join-Path $projectRoot "outputs\evaluation\gamus_six_class_comparison_v1\report.json"
$logPath = Join-Path $projectRoot "outputs\orchestration\gamus_six_class_spatial_refined_v3.log"
$auditRoot = Join-Path $projectRoot "outputs\evaluation\gamus_six_class_spatial_refined_v3"
$experimentPattern = "*_multidomain_surface_gamus_six_class_spatial_refined_v3"

# Frozen experiment inputs and code. Any drift fails closed before GPU work.
$expectedConfigSha256 = "a91acb93ef8a9e6c376ba1b99b71e0a8390ab6132aeda71790e433e9bdf55946"
$expectedSourceConfigSha256 = "97ed2e6c97e4d00b8a6929a06bf1c0821de314d32d3a0f2afa85b119cb99c2ca"
$expectedProtectedSha256 = "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
$expectedPointerSha256 = "a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9"
$expectedSamplingIndexSha256 = "afdd9178c0e744ea20e52f88999130110c3b880dd499e25ea79dff0a3862363b"
$expectedComparisonSha256 = "c487424ddd1bb5032a8a0d3e719254b819623cd64d7940333c53cb9269127e09"
$expectedTrainerSha256 = "bd28ce03fa4acde935701c234b17c2e15f52aa78f478ab118f099d86e7a7e424"
$expectedAuditorSha256 = "d6fa85b87fce3bd7da5221380e2c70f968e63da31716d9536861d759ed88df1f"
$expectedModelCodeSha256 = "a59b97735e4fb57599507d1ca38da30c95a64e8b1d637ed2448120dc4e41b860"
$expectedLossCodeSha256 = "e04a62a7b119cf7f8b76c2a041b512fc456afdbe6c4e19d90a7aeafc806326dd"
$expectedDatasetCodeSha256 = "1aa341f01bb26d34df20af2b854cd9f33c5dacbe7234038e77638f0998347108"
$expectedMetricsCodeSha256 = "b0b0035a1c3db4221482fe92736e6b03615133db08180dab2312f6781fc5874a"

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
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Value))
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

Assert-Hash $configPath $expectedConfigSha256 "frozen spatial-refined v3 config"
Assert-Hash $sourceConfigPath $expectedSourceConfigSha256 "sealed v2 source recipe"
Assert-Hash $protectedCheckpoint $expectedProtectedSha256 "protected production checkpoint"
Assert-Hash $pointerPath $expectedPointerSha256 "live application pointer"
Assert-Hash $samplingIndex $expectedSamplingIndexSha256 "sealed water-sampling index"
Assert-Hash $comparisonReport $expectedComparisonSha256 "v1 comparison report"
Assert-Hash $trainer $expectedTrainerSha256 "frozen multidomain trainer"
Assert-Hash $auditor $expectedAuditorSha256 "frozen v3 checkpoint auditor"
Assert-Hash $modelCode $expectedModelCodeSha256 "frozen spatial-head model code"
Assert-Hash $lossCode $expectedLossCodeSha256 "frozen loss implementation"
Assert-Hash $datasetCode $expectedDatasetCodeSha256 "frozen GAMUS dataset implementation"
Assert-Hash $metricsCode $expectedMetricsCodeSha256 "frozen classification metrics"

$pointerBefore = Get-Content -LiteralPath $pointerPath -Raw
$pointerHashBefore = (Get-FileHash -LiteralPath $pointerPath -Algorithm SHA256).Hash
$pointerTargetBefore = Resolve-ProjectPath ($pointerBefore.Trim())
if ($pointerTargetBefore -ne [System.IO.Path]::GetFullPath($protectedCheckpoint)) {
    throw "The live app pointer no longer targets the protected checkpoint"
}
$protectedHashBefore = (Get-FileHash -LiteralPath $protectedCheckpoint -Algorithm SHA256).Hash

Push-Location $projectRoot
try {
    Write-RunLog "PREFLIGHT spatial-refined six-class v3"
    Write-RunLog "Only fine_semantic_head.* may train; official test and app promotion are forbidden"
    if (-not $TrainOnly) {
        Invoke-LoggedPython -Arguments @(
            "-m", "pytest", "-q",
            "tests\test_gamus_spatial_refined_v3_protocol.py",
            "tests\test_domain_surface_model.py",
            "tests\test_train_multidomain_gamus.py",
            "tests\test_classification_metrics.py"
        )
    }
    if ($PreflightOnly) {
        Write-RunLog "PREFLIGHT COMPLETE; no training was started"
        return
    }

    $activeTrainer = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        $_.ProcessId -ne $PID -and
        $_.CommandLine -match 'train_multidomain\.py'
    }
    if ($activeTrainer) {
        throw "Another Monocular Surface Reconstruction trainer is already active"
    }

    $existing = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
        -Directory -Filter $experimentPattern -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if ($Resume) {
        if (-not $existing -or -not (Test-Path -LiteralPath (Join-Path $existing.FullName "checkpoint_latest.pt"))) {
            throw "-Resume requires an existing spatial-refined v3 checkpoint_latest.pt"
        }
    }
    elseif ($existing) {
        throw "A spatial-refined v3 experiment already exists; use -Resume instead of duplicating it"
    }

    Write-RunLog "START protected-base spatial-refined six-class-head-only experiment"
    $trainingArguments = @(
        "-u", "scripts\train_multidomain.py",
        "--config", $configPath
    )
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
        throw "Trainer exited without a spatial-refined v3 experiment directory"
    }

    $checkpoints = @(
        Join-Path $completed.FullName "checkpoint_latest.pt"
    )
    $best = Join-Path $completed.FullName "checkpoint_best_landscape.pt"
    if (Test-Path -LiteralPath $best -PathType Leaf) {
        $checkpoints += $best
    }
    foreach ($checkpoint in $checkpoints) {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($checkpoint)
        Invoke-LoggedPython -Arguments @(
            "-u", "scripts\audit_gamus_spatial_refined_head_checkpoint.py",
            "--base-checkpoint", $protectedCheckpoint,
            "--candidate-checkpoint", $checkpoint,
            "--source-recipe-config", $sourceConfigPath,
            "--comparison-report", $comparisonReport,
            "--output", (Join-Path $auditRoot "${stem}_audit.json")
        )
    }
    Write-RunLog "COMPLETE spatial-refined v3 and exact inherited-tensor audits"
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

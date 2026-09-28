param(
    [switch]$TrainOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot "configs\multidomain_surface_gamus_six_class_head_only_v2.yaml"
$trainer = Join-Path $projectRoot "scripts\train_multidomain.py"
$auditor = Join-Path $projectRoot "scripts\audit_gamus_head_only_checkpoint.py"
$pointerPath = Join-Path $projectRoot "outputs\runtime\showcase_checkpoint.txt"
$protectedCheckpoint = Join-Path $projectRoot "experiments\20260829T203146Z_multidomain_surface_v2_guarded_vegetation\checkpoint_best_guarded_vegetation.pt"
$samplingIndex = Join-Path $projectRoot "outputs\analysis\gamus_train_six_class_tile_index_v1.jsonl"
$comparisonReport = Join-Path $projectRoot "outputs\evaluation\gamus_six_class_comparison_v1\report.json"
$logPath = Join-Path $projectRoot "outputs\orchestration\gamus_six_class_head_only_v2.log"
$auditRoot = Join-Path $projectRoot "outputs\evaluation\gamus_six_class_head_only_v2"
$experimentPattern = "*_multidomain_surface_gamus_six_class_head_only_v2"

$expectedConfigSha256 = "97ed2e6c97e4d00b8a6929a06bf1c0821de314d32d3a0f2afa85b119cb99c2ca"
$expectedProtectedSha256 = "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
$expectedSamplingIndexSha256 = "afdd9178c0e744ea20e52f88999130110c3b880dd499e25ea79dff0a3862363b"
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

Assert-Hash $configPath $expectedConfigSha256 "frozen head-only config"
Assert-Hash $protectedCheckpoint $expectedProtectedSha256 "protected production checkpoint"
Assert-Hash $samplingIndex $expectedSamplingIndexSha256 "sealed water-sampling index"
Assert-Hash $comparisonReport $expectedComparisonSha256 "v1 comparison report"

$pointerBefore = Get-Content -LiteralPath $pointerPath -Raw
$pointerHashBefore = (Get-FileHash -LiteralPath $pointerPath -Algorithm SHA256).Hash
$pointerTargetBefore = Resolve-ProjectPath ($pointerBefore.Trim())
if ($pointerTargetBefore -ne [System.IO.Path]::GetFullPath($protectedCheckpoint)) {
    throw "The live app pointer no longer targets the protected checkpoint"
}
$protectedHashBefore = (Get-FileHash -LiteralPath $protectedCheckpoint -Algorithm SHA256).Hash

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
        throw "-Resume requires an existing head-only checkpoint_latest.pt"
    }
}
elseif ($existing) {
    throw "A head-only v2 experiment already exists; use -Resume instead of duplicating it"
}

Push-Location $projectRoot
try {
    Write-RunLog "START protected-base six-class-head-only correction"
    Write-RunLog "Only fine_semantic_head is trainable; app promotion is forbidden"
    if (-not $TrainOnly) {
        Invoke-LoggedPython -Arguments @(
            "-m", "pytest", "-q",
            "tests\test_gamus_dataset.py",
            "tests\test_train_multidomain_gamus.py",
            "tests\test_domain_surface_model.py",
            "tests\test_classification_metrics.py",
            "tests\test_audit_gamus_head_only_checkpoint.py"
        )
    }

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
        throw "Trainer exited without a head-only experiment directory"
    }

    $latest = Join-Path $completed.FullName "checkpoint_latest.pt"
    Invoke-LoggedPython -Arguments @(
        "-u", "scripts\audit_gamus_head_only_checkpoint.py",
        "--base-checkpoint", $protectedCheckpoint,
        "--candidate-checkpoint", $latest,
        "--comparison-report", $comparisonReport,
        "--output", (Join-Path $auditRoot "checkpoint_latest_audit.json")
    )
    $best = Join-Path $completed.FullName "checkpoint_best_landscape.pt"
    if (Test-Path -LiteralPath $best -PathType Leaf) {
        Invoke-LoggedPython -Arguments @(
            "-u", "scripts\audit_gamus_head_only_checkpoint.py",
            "--base-checkpoint", $protectedCheckpoint,
            "--candidate-checkpoint", $best,
            "--comparison-report", $comparisonReport,
            "--output", (Join-Path $auditRoot "checkpoint_best_audit.json")
        )
    }
    Write-RunLog "COMPLETE head-only correction and frozen-state audits"
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

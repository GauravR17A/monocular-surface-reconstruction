param(
    [switch]$RunFullValidation
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$calibrator = Join-Path $projectRoot "scripts\calibrate_gamus_six_class_decisions.py"
$protocol = Join-Path $projectRoot "configs\gamus_six_class_spatial_refined_v3_decision_bias_v1.yaml"
$trainingConfig = Join-Path $projectRoot "configs\multidomain_surface_gamus_six_class_spatial_refined_v3.yaml"
$pointer = Join-Path $projectRoot "outputs\runtime\showcase_checkpoint.txt"
$protected = Join-Path $projectRoot "experiments\20260829T203146Z_multidomain_surface_v2_guarded_vegetation\checkpoint_best_guarded_vegetation.pt"
$audit = Join-Path $projectRoot "outputs\evaluation\gamus_six_class_spatial_refined_v3\checkpoint_best_landscape_audit.json"
$outputRoot = Join-Path $projectRoot "outputs\evaluation\gamus_six_class_spatial_refined_v3_decision_bias_dc_phl_v1"
$log = Join-Path $projectRoot "outputs\orchestration\gamus_six_class_spatial_refined_v3_decision_bias_v1.log"
$experimentPattern = "*_multidomain_surface_gamus_six_class_spatial_refined_v3"

$expectedCalibratorSha256 = "8d34fadb0132ae9ccb83755c68203ba32cbee7803b28bcbcf6500e9878d44722"
$expectedProtocolSha256 = "4c6ee2ccb28ccbbd83cacfd8669b8f20e8dd11f578553ecb07f96e828b1a7a87"
$expectedTrainingConfigSha256 = "a91acb93ef8a9e6c376ba1b99b71e0a8390ab6132aeda71790e433e9bdf55946"
$expectedPointerSha256 = "a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9"
$expectedProtectedSha256 = "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
$confirmation = "RUN_FULL_DC_PHL_SPATIAL_V3_DECISION_BIAS_EVALUATION_ONCE"

New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

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

Assert-Hash $calibrator $expectedCalibratorSha256 "sealed decision calibrator"
Assert-Hash $protocol $expectedProtocolSha256 "sealed v3 calibration protocol"
Assert-Hash $trainingConfig $expectedTrainingConfigSha256 "sealed v3 training config"
Assert-Hash $pointer $expectedPointerSha256 "live application pointer"
Assert-Hash $protected $expectedProtectedSha256 "protected production checkpoint"

$activeTrainer = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'train_multidomain\.py'
}
if ($activeTrainer) {
    throw "The v3 trainer is still active; checkpoint selection and audit must finish first"
}

$experiment = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
    -Directory -Filter $experimentPattern -ErrorAction Stop |
    Sort-Object Name -Descending |
    Select-Object -First 1
if (-not $experiment) {
    throw "No completed spatial-refined v3 experiment exists"
}
$checkpoint = Join-Path $experiment.FullName "checkpoint_best_landscape.pt"
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
    throw "The selected v3 checkpoint is missing: $checkpoint"
}
if (-not (Test-Path -LiteralPath $audit -PathType Leaf)) {
    throw "The independent v3 tensor audit has not completed: $audit"
}

$pointerBefore = (Get-FileHash -LiteralPath $pointer -Algorithm SHA256).Hash
$protectedBefore = (Get-FileHash -LiteralPath $protected -Algorithm SHA256).Hash
$checkpointBefore = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash

Push-Location $projectRoot
try {
    Write-RunLog "START bounded DC+PHL decision-bias fit; NYC and official test remain unopened"
    Invoke-Python -Arguments @(
        "-u", $calibrator,
        "--fit",
        "--protocol", $protocol,
        "--candidate-checkpoint", $checkpoint
    )
    $fit = Join-Path $outputRoot "fit.json"
    $fitPayload = Get-Content -LiteralPath $fit -Raw | ConvertFrom-Json
    if ($fitPayload.eligible_for_one_full_dc_phl_evaluation -ne $true) {
        Write-RunLog "STOP sampled calibration gates failed; authoritative full pass is forbidden"
        return
    }
    Write-RunLog "PASS sampled calibration gates"
    if ($RunFullValidation) {
        Invoke-Python -Arguments @(
            "-u", $calibrator,
            "--protocol", $protocol,
            "--candidate-checkpoint", $checkpoint,
            "--fit-artifact", $fit,
            "--confirmation", $confirmation
        )
        Write-RunLog "COMPLETE one authorized full DC+PHL calibration evaluation"
    }
    else {
        Write-RunLog "READY full pass was not consumed; rerun with -RunFullValidation"
    }
}
finally {
    Pop-Location
    $pointerAfter = (Get-FileHash -LiteralPath $pointer -Algorithm SHA256).Hash
    $protectedAfter = (Get-FileHash -LiteralPath $protected -Algorithm SHA256).Hash
    $checkpointAfter = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash
    if (
        $pointerBefore -ne $pointerAfter -or
        $protectedBefore -ne $protectedAfter -or
        $checkpointBefore -ne $checkpointAfter
    ) {
        throw "Protected state changed during v3 decision calibration"
    }
    Write-RunLog "VERIFIED candidate, protected checkpoint, and app pointer are unchanged"
}

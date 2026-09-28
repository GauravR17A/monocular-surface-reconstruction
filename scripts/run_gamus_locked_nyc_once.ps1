param(
    [Parameter(Mandatory = $true)]
    [string]$CandidateCheckpoint,
    [switch]$ConsumeLockedNYC
)

$ErrorActionPreference = "Stop"
if (-not $ConsumeLockedNYC) {
    throw "Refusing to consume NYC without the explicit -ConsumeLockedNYC switch"
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$evaluator = Join-Path $projectRoot "scripts\evaluate_gamus_locked_nyc.py"
$protocol = Join-Path $projectRoot "configs\gamus_locked_nyc_evaluation_v1.yaml"
$pointer = Join-Path $projectRoot "outputs\runtime\showcase_checkpoint.txt"
$protected = Join-Path $projectRoot "experiments\20260829T203146Z_multidomain_surface_v2_guarded_vegetation\checkpoint_best_guarded_vegetation.pt"
$outputRoot = Join-Path $projectRoot "outputs\evaluation\gamus_locked_nyc_v1"

$expectedEvaluatorSha256 = "ef304591d96b93f67abcab5ec841f080b8ee36d8aeb275e8ea847d470c5d5d52"
$expectedProtocolSha256 = "ea24f5f2677c1f82239112ff01254dde6322a4d5f5272e8f9100088bd8029104"
$expectedPointerSha256 = "a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9"
$expectedProtectedSha256 = "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"

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

Assert-Hash $evaluator $expectedEvaluatorSha256 "locked evaluator"
Assert-Hash $protocol $expectedProtocolSha256 "locked evaluation protocol"
Assert-Hash $pointer $expectedPointerSha256 "live pointer"
Assert-Hash $protected $expectedProtectedSha256 "protected checkpoint"

if ([System.IO.Path]::IsPathRooted($CandidateCheckpoint)) {
    $checkpoint = [System.IO.Path]::GetFullPath($CandidateCheckpoint)
}
else {
    $checkpoint = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $CandidateCheckpoint))
}
if ([System.IO.Path]::GetFileName($checkpoint) -ne "checkpoint_best_landscape.pt") {
    throw "Only checkpoint_best_landscape.pt may consume the locked NYC holdout"
}
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
    throw "Candidate checkpoint is missing: $checkpoint"
}
$activeTrainer = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'train_multidomain\.py'
}
if ($activeTrainer) {
    throw "A trainer is active; freeze and model selection must finish before NYC evaluation"
}

$pointerBefore = (Get-FileHash -LiteralPath $pointer -Algorithm SHA256).Hash
$protectedBefore = (Get-FileHash -LiteralPath $protected -Algorithm SHA256).Hash
Push-Location $projectRoot
try {
    & $python -u $evaluator `
        --evaluate-locked `
        --protocol $protocol `
        --candidate-checkpoint $checkpoint `
        --output-root $outputRoot `
        --confirmation "CONSUME_LOCKED_NYC_HOLDOUT_ONCE"
    if ($LASTEXITCODE -ne 0) {
        throw "Locked NYC evaluation failed; the holdout remains consumed and cannot be retried automatically"
    }
}
finally {
    Pop-Location
    $pointerAfter = (Get-FileHash -LiteralPath $pointer -Algorithm SHA256).Hash
    $protectedAfter = (Get-FileHash -LiteralPath $protected -Algorithm SHA256).Hash
    if ($pointerBefore -ne $pointerAfter -or $protectedBefore -ne $protectedAfter) {
        throw "Live application model protection failed during locked evaluation"
    }
}

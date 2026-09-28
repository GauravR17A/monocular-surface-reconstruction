param(
    [switch]$PrepareOnly,
    [switch]$TrainOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "multidomain_v2_pipeline.log"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Invoke-LoggedPython {
    param([string[]]$Arguments)
    "[$(Get-Date -Format o)] python $($Arguments -join ' ')" | Tee-Object -FilePath $logPath -Append
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $commandExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($commandExitCode -ne 0) {
        throw "Monocular Surface Reconstruction command failed with exit code $commandExitCode"
    }
}

Set-Location $projectRoot

if (-not $TrainOnly) {
    Invoke-LoggedPython -Arguments @(
        "scripts\data\prepare_open_canopy_subset.py",
        "data\open_canopy\geometries.geojson",
        "data\open_canopy_v2",
        "--train", "600",
        "--validation", "120",
        "--test", "120",
        "--output-size", "384",
        "--chip-size-m", "576",
        "--min-valid-fraction", "0.50",
        "--min-vegetation-fraction", "0.10",
        "--max-attempt-multiplier", "10",
        "--seed", "20260829"
    )

    Invoke-LoggedPython -Arguments @(
        "scripts\data\export_highbuild_surface_subset.py",
        "data\highbuild_full\msr_splits",
        "data\multidomain_urban_v2",
        "--train-per-region", "90",
        "--validation-per-region", "40",
        "--test-per-region", "40",
        "--seed", "20260829"
    )

    foreach ($split in @("train", "validation", "test")) {
        Invoke-LoggedPython -Arguments @(
            "scripts\data\combine_surface_manifests.py",
            "data\multidomain_v2\manifests\$split.csv",
            "data\multidomain_urban_v2\manifests\$split.csv",
            "data\open_canopy_v2\manifests\$split.csv"
        )
        Invoke-LoggedPython -Arguments @(
            "scripts\data\precompute_surface_priors.py",
            "data\multidomain_v2\manifests\$split.csv",
            "data\multidomain_v2\priors\$split",
            "data\multidomain_v2\manifests_with_priors\$split.csv",
            "--device", "cuda",
            "--model", "models\foundation\depth-anything-v2-small-hf"
        )
    }

    Invoke-LoggedPython -Arguments @("-m", "pytest", "-q")
}

if (-not $PrepareOnly) {
    $latestCheckpoint = Get-ChildItem -Path (Join-Path $projectRoot "experiments") `
        -Directory -Filter "*_multidomain_surface_v2" |
        Sort-Object LastWriteTime -Descending |
        ForEach-Object { Join-Path $_.FullName "checkpoint_latest.pt" } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
    if ($latestCheckpoint) {
        $savedConfig = Join-Path (Split-Path -Parent $latestCheckpoint) "config.yaml"
        Invoke-LoggedPython -Arguments @(
            "scripts\train_multidomain.py",
            "--config", $savedConfig,
            "--resume", $latestCheckpoint
        )
    }
    else {
        Invoke-LoggedPython -Arguments @(
            "scripts\train_multidomain.py",
            "--config", "configs\multidomain_surface_v2.yaml"
        )
    }
}

"[$(Get-Date -Format o)] Multidomain v2 pipeline complete" | Tee-Object -FilePath $logPath -Append

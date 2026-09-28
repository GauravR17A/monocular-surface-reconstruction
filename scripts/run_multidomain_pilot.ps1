param(
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logDirectory = Join-Path $projectRoot "outputs\orchestration"
$logPath = Join-Path $logDirectory "multidomain_pipeline.log"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Invoke-LoggedPython {
    param([string[]]$Arguments)
    "[$(Get-Date -Format o)] python $($Arguments -join ' ')" | Tee-Object -FilePath $logPath -Append
    $previousErrorActionPreference = $ErrorActionPreference
    # tqdm and several geospatial libraries legitimately write progress/warnings
    # to stderr. Reviewer native commands by their exit code, not the stream used.
    $ErrorActionPreference = "Continue"
    & $python @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $commandExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($commandExitCode -ne 0) {
        throw "Monocular Surface Reconstruction command failed with exit code $commandExitCode"
    }
}

Set-Location $projectRoot

Invoke-LoggedPython -Arguments @(
    "scripts\data\prepare_open_canopy_subset.py",
    "data\open_canopy\geometries.geojson",
    "data\open_canopy_pilot",
    "--train", "48",
    "--validation", "12",
    "--test", "12",
    "--output-size", "384",
    "--chip-size-m", "576",
    "--min-valid-fraction", "0.50",
    "--min-vegetation-fraction", "0.10"
)

Invoke-LoggedPython -Arguments @(
    "scripts\data\export_highbuild_surface_subset.py",
    "data\highbuild_full\msr_splits",
    "data\multidomain_urban",
    "--train-per-region", "24",
    "--validation-per-region", "12",
    "--test-per-region", "12"
)

foreach ($split in @("train", "validation", "test")) {
    Invoke-LoggedPython -Arguments @(
        "scripts\data\combine_surface_manifests.py",
        "data\multidomain\manifests\$split.csv",
        "data\multidomain_urban\manifests\$split.csv",
        "data\open_canopy_pilot\manifests\$split.csv"
    )
    Invoke-LoggedPython -Arguments @(
        "scripts\data\precompute_surface_priors.py",
        "data\multidomain\manifests\$split.csv",
        "data\multidomain\priors\$split",
        "data\multidomain\manifests_with_priors\$split.csv",
        "--device", "cuda",
        "--model", "models\foundation\depth-anything-v2-small-hf"
    )
}

Invoke-LoggedPython -Arguments @("-m", "pytest", "-q")

if (-not $PrepareOnly) {
    $latestCheckpoint = Get-ChildItem -Path (Join-Path $projectRoot "experiments") `
        -Directory -Filter "*_multidomain_surface_pilot" |
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
            "--config", "configs\multidomain_surface_pilot.example.yaml"
        )
    }
}

"[$(Get-Date -Format o)] Multidomain pipeline complete" | Tee-Object -FilePath $logPath -Append

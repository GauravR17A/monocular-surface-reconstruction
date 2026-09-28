param(
    [int]$DownloadProcessId = 0,
    [int]$TrainingProcessId = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogRoot = Join-Path $ProjectRoot "outputs\orchestration"
$LogPath = Join-Path $LogRoot "full_pipeline.log"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Set-Location $ProjectRoot

function Write-PipelineLog {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $LogPath -Value $line
}

function Wait-TrackedProcess {
    param([int]$ProcessId, [string]$Label)
    if ($ProcessId -le 0) { return }
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Write-PipelineLog "Waiting for $Label process $ProcessId"
        Wait-Process -Id $ProcessId
    }
    Write-PipelineLog "$Label process is complete"
}

function Invoke-PythonStep {
    param([string]$Label, [string[]]$StepArguments)
    Write-PipelineLog "START $Label"
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @StepArguments *>> $LogPath
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode. See $LogPath"
    }
    Write-PipelineLog "COMPLETE $Label"
}

try {
    Write-PipelineLog "Monocular Surface Reconstruction continuation runner started"
    Wait-TrackedProcess -ProcessId $DownloadProcessId -Label "full dataset download"
    Wait-TrackedProcess -ProcessId $TrainingProcessId -Label "reviewer training"

    Invoke-PythonStep "resumable full download verification" @(
        "scripts\data\download_highbuild_full.py",
        "--output-root", "data\highbuild_full",
        "--include-shards", "--workers", "6"
    )
    Invoke-PythonStep "city and license split preparation" @(
        "scripts\data\prepare_highbuild_full.py"
    )
    Invoke-PythonStep "atomic selected-shard repack" @(
        "scripts\data\repack_highbuild_full.py"
    )
    Invoke-PythonStep "test suite" @("-m", "pytest", "-q")

    $existing = Get-ChildItem (Join-Path $ProjectRoot "experiments") -Directory |
        Where-Object Name -Like "*_highbuild_full_convnext_tiny" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -ne $existing -and (Test-Path (Join-Path $existing.FullName "checkpoint_latest.pt"))) {
        Invoke-PythonStep "resume full-data training" @(
            "scripts\train.py",
            "--config", (Join-Path $existing.FullName "config.yaml"),
            "--resume", (Join-Path $existing.FullName "checkpoint_latest.pt")
        )
    } else {
        try {
            Invoke-PythonStep "start batch-8 full-data training" @(
                "scripts\train.py", "--config", "configs\highbuild_full_train.yaml"
            )
        } catch {
            $recentLog = (Get-Content -LiteralPath $LogPath -Tail 200) -join "`n"
            if ($recentLog -match "CUDA out of memory|CUDA error: out of memory") {
                Write-PipelineLog "Batch 8 exceeded VRAM; retrying safe batch 4 x accumulation 2"
                Invoke-PythonStep "start batch-4 full-data training" @(
                    "scripts\train.py", "--config", "configs\highbuild_full_train_batch4.yaml"
                )
            } else {
                throw
            }
        }
    }
    Write-PipelineLog "Monocular Surface Reconstruction continuation runner completed successfully"
} catch {
    Write-PipelineLog "FAILED: $($_.Exception.Message)"
    throw
}

param(
    [int]$RefreshSeconds = 2
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = Join-Path $projectRoot "data\open_canopy_v2"
$logPath = Join-Path $projectRoot "outputs\orchestration\multidomain_v2_pipeline.log"
$targets = @{ train = 600; validation = 120; test = 120 }
$totalTarget = 840
$existingFiles = @(
    Get-ChildItem -LiteralPath $dataRoot -File -Recurse -ErrorAction SilentlyContinue
)
$previousBytes = [long](($existingFiles | Measure-Object Length -Sum).Sum)
$previousSampleTime = Get-Date
$smoothedBytesPerSecond = 0.0
try { $Host.UI.RawUI.WindowTitle = "Monocular Surface Reconstruction v2 - Live Progress" } catch { }

function Format-Bytes([double]$Bytes) {
    if ($Bytes -ge 1GB) { return "{0:N2} GB" -f ($Bytes / 1GB) }
    return "{0:N1} MB" -f ($Bytes / 1MB)
}

function Format-Duration([double]$Seconds) {
    if ([double]::IsInfinity($Seconds) -or [double]::IsNaN($Seconds) -or $Seconds -lt 0) {
        return "calculating..."
    }
    $duration = [TimeSpan]::FromSeconds($Seconds)
    if ($duration.TotalHours -ge 1) {
        return "{0}h {1}m" -f [math]::Floor($duration.TotalHours), $duration.Minutes
    }
    return "{0}m {1}s" -f $duration.Minutes, $duration.Seconds
}

while ($true) {
    $now = Get-Date
    $sourceFiles = @(
        Get-ChildItem -LiteralPath $dataRoot -Filter source.json -File -Recurse `
            -ErrorAction SilentlyContinue
    )
    $accepted = @{ train = 0; validation = 0; test = 0 }
    foreach ($sourceFile in $sourceFiles) {
        try {
            $metadata = Get-Content -LiteralPath $sourceFile.FullName -Raw | ConvertFrom-Json
            if (
                [double]$metadata.valid_fraction -ge 0.50 -and
                [double]$metadata.vegetation_fraction -ge 0.10
            ) {
                $split = [string]$metadata.official_split
                if ($split -eq "val") { $split = "validation" }
                if ($accepted.ContainsKey($split)) { $accepted[$split]++ }
            }
        }
        catch {
            # A source file may be between its atomic write and rename.
        }
    }

    $acceptedTotal = $accepted.train + $accepted.validation + $accepted.test
    $percent = [math]::Min(100.0, 100.0 * $acceptedTotal / $totalTarget)
    $filled = [math]::Floor($percent / 2)
    $bar = ("#" * $filled).PadRight(50, "-")

    $dataFiles = @(
        Get-ChildItem -LiteralPath $dataRoot -File -Recurse -ErrorAction SilentlyContinue
    )
    $downloadedBytes = [long](($dataFiles | Measure-Object Length -Sum).Sum)
    $sampleSeconds = [math]::Max(($now - $previousSampleTime).TotalSeconds, 0.1)
    $instantBytesPerSecond = [math]::Max(
        0.0,
        ($downloadedBytes - $previousBytes) / $sampleSeconds
    )
    $smoothedBytesPerSecond = if ($smoothedBytesPerSecond -eq 0) {
        $instantBytesPerSecond
    }
    else {
        0.75 * $smoothedBytesPerSecond + 0.25 * $instantBytesPerSecond
    }
    $previousBytes = $downloadedBytes
    $previousSampleTime = $now

    $pipelineProcess = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -match "Monocular Surface Reconstruction" -and
            $_.CommandLine -match "prepare_open_canopy_subset|export_highbuild_surface_subset|precompute_surface_priors|train_multidomain"
        } |
        Select-Object -First 1
    $stage = "Waiting"
    if ($pipelineProcess.CommandLine -match "prepare_open_canopy_subset") {
        $stage = "Downloading Open-Canopy satellite + LiDAR chips"
    }
    elseif ($pipelineProcess.CommandLine -match "export_highbuild_surface_subset") {
        $stage = "Preparing balanced urban chips"
    }
    elseif ($pipelineProcess.CommandLine -match "precompute_surface_priors") {
        $stage = "Computing CUDA relative-depth priors"
    }
    elseif ($pipelineProcess.CommandLine -match "train_multidomain") {
        $stage = "Training guarded multidomain v2 model"
    }

    $elapsedSeconds = 0.0
    if ($pipelineProcess) {
        try {
            $processDetails = Get-Process -Id $pipelineProcess.ProcessId -ErrorAction Stop
            $elapsedSeconds = ($now - $processDetails.StartTime).TotalSeconds
        }
        catch { }
    }
    $chipRate = if ($elapsedSeconds -gt 0) { $acceptedTotal / $elapsedSeconds } else { 0.0 }
    $etaSeconds = if ($chipRate -gt 0) {
        ($totalTarget - $acceptedTotal) / $chipRate
    }
    else {
        [double]::PositiveInfinity
    }

    Clear-Host
    Write-Host "MSR V2 - LIVE DATA PREPARATION" -ForegroundColor Cyan
    Write-Host ""
    Write-Host ("[{0}] {1,5:N1}%" -f $bar, $percent) -ForegroundColor Green
    Write-Host ""
    Write-Host ("Accepted chips : {0} / {1}" -f $acceptedTotal, $totalTarget)
    Write-Host ("  Test          {0,3} / {1}" -f $accepted.test, $targets.test)
    Write-Host ("  Validation    {0,3} / {1}" -f $accepted.validation, $targets.validation)
    Write-Host ("  Train         {0,3} / {1}" -f $accepted.train, $targets.train)
    Write-Host ""
    Write-Host ("Downloaded     : {0}" -f (Format-Bytes $downloadedBytes))
    Write-Host ("Disk write rate: {0}/s" -f (Format-Bytes $smoothedBytesPerSecond))
    Write-Host ("Accepted rate  : {0:N2} chips/min" -f ($chipRate * 60.0))
    Write-Host ("ETA             : {0}" -f (Format-Duration $etaSeconds))
    Write-Host ("Stage           : {0}" -f $stage)
    Write-Host ("Pipeline health : {0}" -f $(if ($pipelineProcess) { "RUNNING" } else { "NOT RUNNING" })) `
        -ForegroundColor $(if ($pipelineProcess) { "Green" } else { "Yellow" })
    Write-Host ("Updated         : {0}" -f $now.ToString("HH:mm:ss"))
    Write-Host ""
    Write-Host "The RTX 4070 becomes active during CUDA-prior generation and training." `
        -ForegroundColor DarkGray
    Write-Host ("Durable log: {0}" -f $logPath) -ForegroundColor DarkGray
    Start-Sleep -Seconds ([math]::Max($RefreshSeconds, 1))
}

param(
    [string]$TargetRoot = "D:\MSRData\GAMUS",
    [ValidateRange(1, 8)]
    [int]$MaxWorkers = 2,
    [ValidateRange(1, 20)]
    [int]$MaxAttempts = 12
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$target = [System.IO.Path]::GetFullPath($TargetRoot)
$allowedRoot = [System.IO.Path]::GetFullPath("D:\MSRData")
if (-not $target.StartsWith(
    $allowedRoot + [System.IO.Path]::DirectorySeparatorChar,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "GAMUS must stay inside $allowedRoot; received $target"
}

$drive = Get-PSDrive -Name D
$minimumFreeBytes = 90GB
if ($drive.Free -lt $minimumFreeBytes -and -not (Test-Path -LiteralPath $target)) {
    throw "At least 90 GiB free is required before the first GAMUS download."
}

$hf = Join-Path $projectRoot ".venv\Scripts\hf.exe"
if (-not (Test-Path -LiteralPath $hf)) {
    throw "The project Hugging Face CLI is missing: $hf"
}

$cacheRoot = Join-Path $allowedRoot ".hf_cache"
$env:HF_HOME = $cacheRoot
$env:HF_XET_CACHE = Join-Path $cacheRoot "xet"
$env:HF_HUB_CACHE = Join-Path $cacheRoot "hub"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"

New-Item -ItemType Directory -Path $target -Force | Out-Null
New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null

Write-Host "Monocular Surface Reconstruction GAMUS download"
Write-Host "Repository : earthflow/GAMUS"
Write-Host "License    : CC-BY-4.0"
Write-Host "Target     : $target"
Write-Host "Cache      : $cacheRoot"
Write-Host "Free D     : $([math]::Round($drive.Free / 1GB, 2)) GiB"

$downloadSucceeded = $false
for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    Write-Host "Download attempt $attempt/$MaxAttempts (resumable, $MaxWorkers workers)"
    & $hf download earthflow/GAMUS `
        --repo-type dataset `
        --local-dir $target `
        --max-workers $MaxWorkers
    if ($LASTEXITCODE -eq 0) {
        $downloadSucceeded = $true
        break
    }

    if ($attempt -lt $MaxAttempts) {
        $retryDelaySeconds = [Math]::Min(60, 10 * $attempt)
        Write-Warning "Download paused with exit code $LASTEXITCODE. Existing files are safe; retrying in $retryDelaySeconds seconds."
        Start-Sleep -Seconds $retryDelaySeconds
    }
}
if (-not $downloadSucceeded) {
    throw "GAMUS download did not complete after $MaxAttempts resumable attempts. Existing files remain intact."
}

$provenance = [ordered]@{
    dataset = "GAMUS"
    repository = "https://huggingface.co/datasets/earthflow/GAMUS"
    official_code = "https://github.com/EarthNets/RSI-MMSegmentation"
    license = "CC-BY-4.0"
    downloaded_at = [DateTime]::UtcNow.ToString("o")
    target = $target
}
$provenance | ConvertTo-Json | Set-Content `
    -LiteralPath (Join-Path $target "msr_provenance.json") `
    -Encoding UTF8

Write-Host "GAMUS download complete: $target" -ForegroundColor Green

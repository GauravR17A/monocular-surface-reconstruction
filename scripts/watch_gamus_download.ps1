param(
    [string]$TargetRoot = "D:\MSRData\GAMUS",
    [double]$TotalBytes = 80027447350,
    [int]$RefreshSeconds = 5
)

$ErrorActionPreference = "Stop"
$target = [System.IO.Path]::GetFullPath($TargetRoot)
$allowed = [System.IO.Path]::GetFullPath("D:\MSRData")
if (-not $target.StartsWith(
    $allowed + [System.IO.Path]::DirectorySeparatorChar,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "GAMUS watcher only accepts a target below $allowed"
}
if ($TotalBytes -le 0 -or $RefreshSeconds -lt 1) {
    throw "TotalBytes and RefreshSeconds must be positive"
}

$previousBytes = 0.0
$previousTime = Get-Date
$smoothedBytesPerSecond = 0.0
$barWidth = 48

while ($true) {
    $now = Get-Date
    $files = Get-ChildItem -LiteralPath $target -File -Recurse -ErrorAction SilentlyContinue
    $downloadedBytes = [double](($files | Measure-Object -Property Length -Sum).Sum)
    $elapsed = [Math]::Max(($now - $previousTime).TotalSeconds, 0.001)
    $instantSpeed = [Math]::Max(($downloadedBytes - $previousBytes) / $elapsed, 0.0)
    if ($previousBytes -gt 0) {
        $smoothedBytesPerSecond = if ($smoothedBytesPerSecond -gt 0) {
            0.70 * $smoothedBytesPerSecond + 0.30 * $instantSpeed
        } else {
            $instantSpeed
        }
    }
    $previousBytes = $downloadedBytes
    $previousTime = $now

    $fraction = [Math]::Min([Math]::Max($downloadedBytes / $TotalBytes, 0.0), 1.0)
    $filled = [Math]::Floor($fraction * $barWidth)
    $bar = ("#" * $filled) + ("-" * ($barWidth - $filled))
    $remainingBytes = [Math]::Max($TotalBytes - $downloadedBytes, 0.0)
    $eta = if ($smoothedBytesPerSecond -gt 0) {
        [TimeSpan]::FromSeconds($remainingBytes / $smoothedBytesPerSecond).ToString("dd\.hh\:mm\:ss")
    } else {
        "calculating..."
    }

    $active = [bool](Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'hf(\.exe)?[\" ]+download[\" ]+earthflow/GAMUS'
    })
    $complete = Test-Path -LiteralPath (Join-Path $target "msr_provenance.json")
    $state = if ($complete) { "COMPLETE" } elseif ($active) { "DOWNLOADING" } else { "RETRYING / CHECKING" }

    Clear-Host
    Write-Host "Monocular Surface Reconstruction - GAMUS dataset" -ForegroundColor Cyan
    Write-Host ""
    Write-Host ("[{0}] {1,6:N2}%" -f $bar, ($fraction * 100)) -ForegroundColor Green
    Write-Host ("Downloaded : {0:N2} / {1:N2} GiB" -f ($downloadedBytes / 1GB), ($TotalBytes / 1GB))
    Write-Host ("Files      : {0:N0} / 26,172 data files" -f $files.Count)
    Write-Host ("Speed      : {0:N2} MiB/s" -f ($smoothedBytesPerSecond / 1MB))
    Write-Host ("ETA        : {0}" -f $eta)
    Write-Host ("Status     : {0}" -f $state)
    Write-Host ("Updated    : {0}" -f $now.ToString("HH:mm:ss"))
    Write-Host ""
    Write-Host "Safe to close this status window; the downloader runs separately." -ForegroundColor DarkGray

    if ($complete) {
        Write-Host "GAMUS is ready. Monocular Surface Reconstruction can begin the full-data training stage." -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds $RefreshSeconds
}

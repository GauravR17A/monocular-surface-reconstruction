param(
    [string]$OutputRoot = "data\highbuild_full",
    [double]$ExpectedGB = 35.0,
    [int]$PollSeconds = 5
)

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$host.UI.RawUI.WindowTitle = "Monocular Surface Reconstruction - Full Dataset Download"
$expectedBytes = $ExpectedGB * 1GB
$previousTransferBytes = 0
$smoothedSpeed = 0
$previousTime = Get-Date

while ($true) {
    $now = Get-Date
    $files = Get-ChildItem -LiteralPath $OutputRoot -Recurse -File -ErrorAction SilentlyContinue
    $bytes = [double](($files | Measure-Object Length -Sum).Sum)
    $elapsed = [math]::Max(($now - $previousTime).TotalSeconds, 0.1)
    $percent = [math]::Min(($bytes / $expectedBytes) * 100, 100)
    $filled = [math]::Min([math]::Floor($percent / 2), 50)
    $bar = ("#" * $filled) + ("-" * (50 - $filled))
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and $_.CommandLine -like "*download_highbuild_full.py*"
    })
    $transferBytes = [double](($active | Measure-Object WriteTransferCount -Sum).Sum)
    $instantSpeed = if ($previousTransferBytes -gt 0 -and $transferBytes -ge $previousTransferBytes) {
        ($transferBytes - $previousTransferBytes) / $elapsed
    } else { 0 }
    if ($instantSpeed -gt 0) {
        $smoothedSpeed = if ($smoothedSpeed -gt 0) {
            (0.6 * $instantSpeed) + (0.4 * $smoothedSpeed)
        } else { $instantSpeed }
    } else {
        $smoothedSpeed *= 0.75
    }
    $eta = if ($smoothedSpeed -gt 0.1MB -and $bytes -lt $expectedBytes) {
        [TimeSpan]::FromSeconds(($expectedBytes - $bytes) / $smoothedSpeed).ToString("hh\:mm\:ss")
    } else { "waiting for data..." }
    $status = if (-not $active) {
        "not running"
    } elseif ($instantSpeed -gt 0.1MB) {
        "receiving data"
    } else {
        "connected / waiting for next chunk"
    }

    Clear-Host
    Write-Host "Monocular Surface Reconstruction full dataset download" -ForegroundColor Cyan
    Write-Host ""
    Write-Host ("[{0}] {1,5:N1}%" -f $bar, $percent) -ForegroundColor Green
    Write-Host ("Downloaded : {0:N2} / ~{1:N1} GB" -f ($bytes / 1GB), $ExpectedGB)
    Write-Host ("Live speed : {0:N1} MB/s" -f ($smoothedSpeed / 1MB))
    Write-Host ("Status     : {0}" -f $status)
    Write-Host ("ETA        : {0}" -f $eta)
    Write-Host ("Updated    : {0}" -f $now.ToString("HH:mm:ss"))
    Write-Host ""
    Write-Host "After this: verify -> prepare splits -> start RTX 4070 training"

    if (-not $active) {
        Write-Host ""
        Write-Host "Download process finished. Dataset preparation will begin automatically." -ForegroundColor Yellow
        break
    }
    $previousTransferBytes = $transferBytes
    $previousTime = $now
    Start-Sleep -Seconds $PollSeconds
}

Write-Host "You may leave this window open."

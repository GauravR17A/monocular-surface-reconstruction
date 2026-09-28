param([int]$RefreshSeconds = 3)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$experimentRoot = Join-Path $projectRoot "experiments"
$ErrorActionPreference = "Stop"
$lastDisplay = ""

function Read-SharedStatusText([string]$Path) {
    # Permit the trainer's atomic rename while this viewer reads the old file.
    $share = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, $share)
    $reader = [System.IO.StreamReader]::new($stream)
    try { return $reader.ReadToEnd() }
    finally { $reader.Dispose() }
}

while ($true) {
    $experiment = Get-ChildItem -LiteralPath $experimentRoot -Directory -Filter "*_residual_height_v1" |
        Sort-Object Name -Descending | Select-Object -First 1
    $display = "Monocular Surface Reconstruction | waiting for residual-height experiment"
    if ($experiment) {
        $statusPath = Join-Path $experiment.FullName "status.json"
        try {
            if (Test-Path -LiteralPath $statusPath) {
                $status = Read-SharedStatusText $statusPath | ConvertFrom-Json
                $display = "Monocular Surface Reconstruction | $($status.stage) | epoch $($status.epoch)"
                if ($status.suite) { $display += " | $($status.suite)" }
                if ($status.total_batches -gt 0) {
                    $completed = if ($null -ne $status.batch) { $status.batch } else { $status.completed_batches }
                    $fraction = [Math]::Min(1.0, $completed / $status.total_batches)
                    $filled = [int][Math]::Floor(30 * $fraction)
                    $bar = ("#" * $filled) + ("-" * (30 - $filled))
                    $display += " [$bar] $completed/$($status.total_batches)"
                    if ($null -ne $status.loss) { $display += " | loss $([Math]::Round($status.loss, 2))" }
                }
                if ($status.error) { $display += " | $($status.error)" }
                $display += " | updated $($status.updated_at_utc)"
                $metricsPath = Join-Path $experiment.FullName "metrics.jsonl"
                if (Test-Path -LiteralPath $metricsPath) {
                    $metricLines = (Read-SharedStatusText $metricsPath) -split "`r?`n" | Where-Object { $_.Trim() }
                    $latest = $metricLines[-1] | ConvertFrom-Json
                    $urban = $latest.validation.highbuild.metrics.domains.building.rmse_m
                    $vegetation = $latest.validation.open_canopy.metrics.domains.vegetation.rmse_m
                    $ground = $latest.validation.open_canopy.metrics.domains.ground.rmse_m
                    $display += "`nLast epoch $($latest.epoch) | RMSE: buildings $([Math]::Round($urban, 2)) m | vegetation $([Math]::Round($vegetation, 2)) m | forest ground $([Math]::Round($ground, 2)) m | guards: $($latest.gate.all_regression_guards_pass)"
                }
            }
        }
        catch {
            $display = "Monocular Surface Reconstruction | waiting for next complete status write | $($experiment.Name)"
        }
    }
    if ($display -ne $lastDisplay) {
        Clear-Host
        Write-Host $display -ForegroundColor Cyan
        Write-Host "Development experiment. Production model unchanged. Ctrl+C closes this viewer only." -ForegroundColor DarkGray
        $lastDisplay = $display
    }
    Start-Sleep -Seconds ([Math]::Max(1, $RefreshSeconds))
}

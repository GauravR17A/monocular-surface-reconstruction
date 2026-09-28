param([int]$RefreshSeconds = 3)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$activePath = Join-Path $projectRoot "outputs\orchestration\rgb_segmenter_v1_active.json"
$ErrorActionPreference = "Stop"
$lastDisplay = ""

function Read-SharedStatusText([string]$Path) {
    $share = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, $share)
    $reader = [System.IO.StreamReader]::new($stream)
    try { return $reader.ReadToEnd() }
    finally { $reader.Dispose() }
}

while ($true) {
    $display = "Monocular Surface Reconstruction | preparing independent six-class training"
    try {
        if (Test-Path -LiteralPath $activePath) {
            $active = Read-SharedStatusText $activePath | ConvertFrom-Json
            $experimentPath = $active.experiment_dir
            if ($experimentPath) {
                $statusPath = Join-Path $experimentPath "status.json"
                if (Test-Path -LiteralPath $statusPath) {
                    $status = Read-SharedStatusText $statusPath | ConvertFrom-Json
                    $display = "Monocular Surface Reconstruction six-class | $($status.stage) | epoch $($status.epoch)"
                    if ($status.total_batches -gt 0) {
                        $completed = if ($null -ne $status.batch) { $status.batch } else { $status.completed_batches }
                        $fraction = [Math]::Max(0, [Math]::Min(1, $completed / $status.total_batches))
                        $filled = [int][Math]::Floor(25 * $fraction)
                        $bar = ("#" * $filled) + ("-" * (25 - $filled))
                        $display += " [$bar] $completed/$($status.total_batches)"
                        if ($null -ne $status.loss) { $display += " | loss $([Math]::Round($status.loss, 3))" }
                    }
                    if ($status.error) { $display += " | $($status.error)" }
                    $metricsPath = Join-Path $experimentPath "metrics.jsonl"
                    if (Test-Path -LiteralPath $metricsPath) {
                        $rows = (Read-SharedStatusText $metricsPath) -split "`r?`n" | Where-Object { $_.Trim() }
                        if ($rows.Count -gt 0) {
                            $latest = $rows[-1] | ConvertFrom-Json
                            $scores = $latest.validation.overall.six_class_identification
                            if ($null -eq $scores) { $scores = $latest.validation.six_class_identification }
                            if ($null -ne $scores.macro_f1) {
                                $display += "`nEpoch $($latest.epoch) F1 $([Math]::Round(100*$scores.macro_f1,1))%"
                                foreach ($name in @('buildings','trees','low_vegetation','roads','water','ground')) {
                                    $display += " | ${name}: $([Math]::Round(100*$scores.per_class.$name.f1,1))"
                                }
                                $display += " | accepted: $($latest.gate.passes)"
                            }
                        }
                    }
                }
            }
        }
    }
    catch { $display = "Monocular Surface Reconstruction | waiting for complete status update" }
    if ($display -ne $lastDisplay) {
        Clear-Host
        Write-Host $display -ForegroundColor Cyan
        Write-Host "Development results only. App/height model unchanged. Ctrl+C closes viewer only." -ForegroundColor DarkGray
        $lastDisplay = $display
    }
    Start-Sleep -Seconds ([Math]::Max(1, $RefreshSeconds))
}

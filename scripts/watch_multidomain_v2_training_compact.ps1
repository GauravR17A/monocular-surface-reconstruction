param([int]$RefreshMilliseconds = 750)

$projectRoot = Split-Path -Parent $PSScriptRoot
$logPath = Join-Path $projectRoot "outputs\orchestration\multidomain_v2_pipeline.log"
$nvidiaSmi = "C:\Windows\System32\nvidia-smi.exe"
$totalEpochs = 24

try { $Host.UI.RawUI.WindowTitle = "Monocular Surface Reconstruction v2 - Compact Training" } catch { }

while ($true) {
    $epoch = 1
    $resultText = "no completed epoch yet"
    $experiment = Get-ChildItem (Join-Path $projectRoot "experiments") -Directory -Filter "*_multidomain_surface_v2" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($experiment) {
        $metricsPath = Join-Path $experiment.FullName "metrics.jsonl"
        if (Test-Path -LiteralPath $metricsPath) {
            $lastMetric = Get-Content -LiteralPath $metricsPath -Tail 1 -ErrorAction SilentlyContinue
            if ($lastMetric) {
                try {
                    $metric = $lastMetric | ConvertFrom-Json
                    $completedEpoch = [int]$metric.epoch
                    $epoch = $completedEpoch + 1
                    $overallRmse = [double]$metric.validation_metrics.rmse_m
                    $forestRmse = [double]$metric.validation_metrics.landscapes.forest.rmse_m
                    $vegetationRmse = [double]$metric.validation_metrics.domains.vegetation.rmse_m
                    $guard = if ($metric.validation_metrics.passes_urban_regression_guard) { "PASS" } else { "FAIL" }
                    $resultText = "E$completedEpoch results O:{0:N2}m F:{1:N2}m V:{2:N2}m G:$guard" -f $overallRmse, $forestRmse, $vegetationRmse
                } catch { }
            }
        }
    }

    $stage = "starting"
    $percent = 0
    $current = 0
    $total = 0
    $loss = $null
    if (Test-Path -LiteralPath $logPath) {
        $recent = Get-Content -LiteralPath $logPath -Tail 350 -ErrorAction SilentlyContinue
        $progressLine = $recent |
            Where-Object { $_ -match "multidomain (train|validation):" } |
            Select-Object -Last 1
        if ($progressLine -match "multidomain (train|validation):\s+(\d+)%.*?([0-9]+)/([0-9]+)") {
            $stage = $Matches[1]
            $percent = [int]$Matches[2]
            $current = [int]$Matches[3]
            $total = [int]$Matches[4]
        }
        if ($progressLine -match "loss=([0-9]+(?:\.[0-9]+)?)") {
            $loss = [double]$Matches[1]
        }
    }

    $filled = [math]::Floor($percent / 5)
    $bar = ("#" * $filled).PadRight(20, "-")
    $gpuText = "RTX waiting"
    if (Test-Path -LiteralPath $nvidiaSmi) {
        $gpu = & $nvidiaSmi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null
        if ($gpu) {
            $parts = $gpu -split ",\s*"
            $gpuText = "RTX $($parts[0])% | VRAM $($parts[1])/$($parts[2])MB | $($parts[3])C"
        }
    }
    $lossText = if ($null -ne $loss) { " | loss {0:N2}" -f $loss } else { "" }
    $line = "[$bar] epoch $epoch/$totalEpochs | $stage $current/$total ($percent%)$lossText | $resultText | $gpuText"
    $width = 150
    try { $width = [math]::Max(40, $Host.UI.RawUI.WindowSize.Width - 1) } catch { }
    if ($line.Length -gt $width) { $line = $line.Substring(0, $width) }
    Write-Host ("`r" + $line.PadRight($width)) -NoNewline
    Start-Sleep -Milliseconds $RefreshMilliseconds
}

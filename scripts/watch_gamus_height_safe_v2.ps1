param(
    [int]$RefreshMilliseconds = 1000,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logPath = Join-Path $projectRoot "outputs\orchestration\gamus_height_safe_v2.log"
$nvidiaSmi = "C:\Windows\System32\nvidia-smi.exe"
$totalEpochs = 4
$experimentPattern = "*_multidomain_surface_gamus_height_safe_v2"

do {
    $experiment = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
        -Directory -Filter $experimentPattern -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1
    $result = "baseline/preflight"
    $nextEpoch = 1
    if ($experiment) {
        $metricsPath = Join-Path $experiment.FullName "metrics.jsonl"
        if (Test-Path -LiteralPath $metricsPath -PathType Leaf) {
            $lastLine = Get-Content -LiteralPath $metricsPath -Tail 1
            if ($lastLine) {
                $metric = $lastLine | ConvertFrom-Json
                $completedEpoch = [int]$metric.epoch
                $nextEpoch = [math]::Min($completedEpoch + 1, $totalEpochs)
                $values = $metric.validation_metrics
                $guard = if ($values.passes_validation_guards) { "PASS" } else { "FAIL" }
                $result = "E$completedEpoch RMSE:{0:N2} MAE:{1:N2} G:{2:N2} B:{3:N2} V:{4:N2} gate:$guard" -f `
                    [double]$values.rmse_m,
                    [double]$values.mae_m,
                    [double]$values.domains.ground.rmse_m,
                    [double]$values.domains.building.rmse_m,
                    [double]$values.domains.vegetation.rmse_m
            }
        }
    }

    $stage = "waiting"
    $percent = 0
    $current = 0
    $total = 0
    $terminalState = $null
    if (Test-Path -LiteralPath $logPath -PathType Leaf) {
        $recent = Get-Content -LiteralPath $logPath -Tail 400 -ErrorAction SilentlyContinue
        $progressLine = $recent |
            Where-Object { $_ -match "multidomain (train|validation):" } |
            Select-Object -Last 1
        if ($progressLine -match "multidomain (train|validation):\s+(\d+)%.*?([0-9]+)/([0-9]+)") {
            $stage = $Matches[1]
            $percent = [int]$Matches[2]
            $current = [int]$Matches[3]
            $total = [int]$Matches[4]
        }
        if ($recent -match "COMPLETE offline GAMUS height-safe v2") {
            $terminalState = "complete"
        }
        elseif ($recent -match "FAILED ") {
            $terminalState = "failed"
        }
    }

    $filled = [math]::Floor($percent / 5)
    $bar = ("#" * $filled).PadRight(20, "-")
    $gpuText = "GPU idle"
    if (Test-Path -LiteralPath $nvidiaSmi -PathType Leaf) {
        $gpu = & $nvidiaSmi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null
        if ($gpu) {
            $parts = $gpu -split ",\s*"
            $gpuText = "GPU $($parts[0])% VRAM $($parts[1])/$($parts[2])MB $($parts[3])C"
        }
    }
    $stateText = if ($terminalState) { " $terminalState" } else { "" }
    $line = "[$bar] epoch $nextEpoch/$totalEpochs $stage $current/$total $percent% | $result | $gpuText$stateText"
    if ($Once) {
        Write-Output $line
    }
    else {
        Write-Host ("`r" + $line.PadRight(150)) -NoNewline
    }
    if ($Once -or $terminalState) { break }
    Start-Sleep -Milliseconds $RefreshMilliseconds
} while ($true)

if (-not $Once) { Write-Host }

param(
    [int]$RefreshMilliseconds = 1000,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = Join-Path $projectRoot "configs\multidomain_surface_gamus_six_class_candidate_v1.yaml"
$expectedConfigSha256 = "6213accf5334ec1d7cfb394fc6322810242af0d5bab3dc7be3ebba8584b8d01a"
$pointerPath = Join-Path $projectRoot "outputs\runtime\showcase_checkpoint.txt"
$expectedPointerSha256 = "a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9"
$protectedCheckpoint = Join-Path $projectRoot "experiments\20260829T203146Z_multidomain_surface_v2_guarded_vegetation\checkpoint_best_guarded_vegetation.pt"
$expectedProtectedSha256 = "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
$logPath = Join-Path $projectRoot "outputs\orchestration\gamus_six_class_candidate_v1.log"
$nvidiaSmi = "C:\Windows\System32\nvidia-smi.exe"
$totalEpochs = 6
$experimentPattern = "*_multidomain_surface_gamus_six_class_candidate_v1"

foreach ($item in @(
    @($configPath, $expectedConfigSha256, "candidate config"),
    @($pointerPath, $expectedPointerSha256, "live pointer"),
    @($protectedCheckpoint, $expectedProtectedSha256, "protected checkpoint")
)) {
    if (-not (Test-Path -LiteralPath $item[0] -PathType Leaf)) {
        throw "$($item[2]) is missing: $($item[0])"
    }
    $actual = (Get-FileHash -LiteralPath $item[0] -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $item[1]) {
        throw "$($item[2]) hash mismatch: $actual"
    }
}

try { $Host.UI.RawUI.WindowTitle = "Monocular Surface Reconstruction - GAMUS six-class candidate" } catch { }

do {
    # Recheck the two production safety artifacts continuously. The monitor
    # stops visibly if anything attempts to replace the live model.
    $pointerHash = (Get-FileHash -LiteralPath $pointerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $protectedHash = (Get-FileHash -LiteralPath $protectedCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($pointerHash -ne $expectedPointerSha256 -or $protectedHash -ne $expectedProtectedSha256) {
        throw "Production checkpoint protection failed while monitoring"
    }

    $experiment = Get-ChildItem -LiteralPath (Join-Path $projectRoot "experiments") `
        -Directory -Filter $experimentPattern -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1
    $result = "no completed epoch"
    $nextEpoch = 1
    if ($experiment) {
        $metricsPath = Join-Path $experiment.FullName "metrics.jsonl"
        if (Test-Path -LiteralPath $metricsPath -PathType Leaf) {
            $lastLine = Get-Content -LiteralPath $metricsPath -Tail 1
            if ($lastLine) {
                try {
                    $metric = $lastLine | ConvertFrom-Json
                    $completedEpoch = [int]$metric.epoch
                    $nextEpoch = [math]::Min($completedEpoch + 1, $totalEpochs)
                    $values = $metric.validation_metrics
                    $guard = if ($values.passes_validation_guards) { "PASS" } else { "FAIL" }
                    $six = $values.six_class_identification
                    $result = "E$completedEpoch H:{0:N2} B:{1:N2} V:{2:N2} | 6F1:{3:N3} WIoU:{4:N3} RIoU:{5:N3} RB:{6:N3} gate:$guard" -f `
                        [double]$values.rmse_m,
                        [double]$values.domains.building.rmse_m,
                        [double]$values.domains.vegetation.rmse_m,
                        [double]$six.macro_f1,
                        [double]$six.per_class.water.iou,
                        [double]$six.per_class.roads.iou,
                        [double]$values.road_boundary_quality.f1
                }
                catch {
                    $result = "latest epoch metrics are incomplete"
                }
            }
        }
    }

    $stage = "waiting"
    $percent = 0
    $current = 0
    $total = 0
    $loss = $null
    $terminalState = $null
    if (Test-Path -LiteralPath $logPath -PathType Leaf) {
        $recent = @(Get-Content -LiteralPath $logPath -Tail 500 -ErrorAction SilentlyContinue)
        $sessionStart = -1
        for ($index = $recent.Count - 1; $index -ge 0; $index--) {
            if ($recent[$index] -match "\] (START|RESUME)\b") {
                $sessionStart = $index
                break
            }
        }
        $session = if ($sessionStart -ge 0) {
            @($recent[$sessionStart..($recent.Count - 1)])
        }
        else {
            @($recent)
        }
        $progressLine = $session |
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
        if ($session -match "COMPLETE offline GAMUS six-class candidate") {
            $terminalState = "complete"
        }
        elseif ($session -match "FAILED ") {
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
    $lossText = if ($null -ne $loss) { " loss:{0:N2}" -f $loss } else { "" }
    $stateText = if ($terminalState) { " $terminalState" } else { "" }
    $line = "[$bar] epoch $nextEpoch/$totalEpochs $stage $current/$total $percent%$lossText | $result | $gpuText$stateText"
    if ($Once) {
        Write-Output $line
    }
    else {
        $width = 180
        try { $width = [math]::Max(40, $Host.UI.RawUI.WindowSize.Width - 1) } catch { }
        if ($line.Length -gt $width) { $line = $line.Substring(0, $width) }
        Write-Host ("`r" + $line.PadRight($width)) -NoNewline
    }

    if ($Once -or $terminalState) {
        break
    }
    Start-Sleep -Milliseconds $RefreshMilliseconds
} while ($true)

if (-not $Once) { Write-Host }

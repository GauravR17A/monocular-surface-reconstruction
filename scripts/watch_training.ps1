param([int]$RefreshSeconds = 2)

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$NvidiaSmi = "C:\Windows\System32\nvidia-smi.exe"
Set-Location $ProjectRoot

while ($true) {
    $experiment = Get-ChildItem (Join-Path $ProjectRoot "experiments") -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "metrics.jsonl") } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    $record = $null
    if ($null -ne $experiment) {
        $line = Get-Content (Join-Path $experiment.FullName "metrics.jsonl") -Tail 1 -ErrorAction SilentlyContinue
        if ($line) { $record = $line | ConvertFrom-Json }
    }
    $gpu = $null
    if (Test-Path $NvidiaSmi) {
        $gpu = & $NvidiaSmi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu --format=csv,noheader,nounits
    }
    $pipelineLog = Join-Path $ProjectRoot "outputs\orchestration\full_pipeline.log"
    $pipeline = if (Test-Path $pipelineLog) { Get-Content $pipelineLog -Tail 1 } else { "Pipeline log not created yet" }

    Clear-Host
    Write-Host "MSR LIVE TRAINING" -ForegroundColor Cyan
    Write-Host "Updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Pipeline: $pipeline"
    if ($null -ne $experiment) {
        Write-Host "Experiment: $($experiment.Name)"
    }
    if ($null -ne $record) {
        Write-Host ""
        Write-Host ("Epoch: {0}    Learning rate: {1:N8}" -f $record.epoch, $record.learning_rate) -ForegroundColor Yellow
        Write-Host ("Validation RMSE:          {0:N3} m" -f $record.validation_metrics.rmse_m)
        Write-Host ("Validation MAE:           {0:N3} m" -f $record.validation_metrics.mae_m)
        Write-Host ("Validation bias:          {0:N3} m" -f $record.validation_metrics.bias_m)
        Write-Host ("Building-only RMSE:       {0:N3} m" -f $record.validation_metrics.building_rmse_m)
        Write-Host ("Correlation:              {0:N3}" -f $record.validation_metrics.correlation)
        Write-Host ("Train total loss:         {0:N3}" -f $record.train_loss.total)
    } else {
        Write-Host "Waiting for the first completed epoch..." -ForegroundColor Yellow
    }
    Write-Host ""
    if ($gpu) {
        $fields = $gpu -split ",\s*"
        Write-Host "RTX 4070" -ForegroundColor Green
        Write-Host ("Compute: {0}%    VRAM: {1}/{2} MiB    Power: {3} W" -f $fields[0], $fields[1], $fields[2], $fields[3])
        Write-Host ("Clock: {0} MHz    Temperature: {1} C" -f $fields[4], $fields[5])
    }
    Write-Host ""
    Write-Host "This window refreshes every $RefreshSeconds seconds. Press Ctrl+C to close it."
    Start-Sleep -Seconds $RefreshSeconds
}

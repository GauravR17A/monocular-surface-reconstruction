param([switch]$Once)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$Host.UI.RawUI.WindowTitle = 'Monocular Surface Reconstruction - Classification training'
function Read-SharedJson([string]$Path) {
    $stream = [System.IO.File]::Open($Path, 'Open', 'Read', ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    try { $reader = [System.IO.StreamReader]::new($stream); return ($reader.ReadToEnd() | ConvertFrom-Json) }
    finally { if ($reader) { $reader.Dispose() }; $stream.Dispose() }
}
do {
    try {
        $activePath = Join-Path $projectRoot 'outputs\orchestration\rgb_multiregion_v3_active.json'
        $metricsLine = ''
        if (Test-Path -LiteralPath $activePath) {
            $run = Read-SharedJson $activePath
            $state = Read-SharedJson (Join-Path $run.experiment 'status.json')
            $done = 0; $total = 1
            if ($state.stage -eq 'training') { $done=$state.batch; $total=$state.total_batches }
            elseif ($state.stage -eq 'oem_validation') { $done=$state.completed; $total=$state.total }
            elseif ($state.stage -eq 'gamus_validation') { $done=$state.completed_batches; $total=$state.total_batches }
            elseif ($state.stage -in @('complete','epoch_complete')) { $done=1 }
            $label = "Epoch $($state.epoch) | $($state.stage)"
            $commitPath = Join-Path $run.experiment 'commit.json'
            if (Test-Path -LiteralPath $commitPath) {
                $commit = Read-SharedJson $commitPath
                $historyPath = Join-Path $run.experiment 'metrics.jsonl'
                if (Test-Path -LiteralPath $historyPath) {
                    $last = Get-Content -LiteralPath $historyPath -Tail 1 | ConvertFrom-Json
                    $gamus = $last.validation.gamus.overall.six_class_identification.macro_f1 * 100
                    $oem = $last.validation.oem.overall.six_class_identification.macro_f1 * 100
                    $metricsLine = 'Last epoch {0}: GAMUS F1 {1:N1}% | OEM F1 {2:N1}% | safety {3} | benefit {4}' -f $last.epoch,$gamus,$oem,$last.gate.safety_pass,$last.gate.benefit_pass
                }
            }
            $detail = if ($state.error) { $state.error } else { 'Classification only | App heights unchanged | Auto-saved every epoch' }
        } else {
            $state = Read-SharedJson 'D:\MSRData\training\oem_multiregion_v3\status.json'
            $done=$state.downloaded_file_bytes; $total=$state.total_file_bytes
            $label='Preparing multi-region data'
            $detail = '{0}/{1} pairs | {2:N1}/{3:N1} MB | {4}' -f $state.pairs,$state.total_pairs,($done/1e6),($total/1e6),$state.stage
            if ($state.error) { $detail += " | $($state.error)" }
        }
        $fraction = [Math]::Min(1.0,[Math]::Max(0.0,([double]$done/[Math]::Max(1.0,[double]$total))))
        $filled=[int]($fraction*35)
        $bar='['+('#'*$filled)+('-'*(35-$filled))+']'
        if (-not $Once) { Clear-Host }
        Write-Host ('{0} {1} {2:N1}%' -f $label,$bar,($fraction*100)) -ForegroundColor Cyan
        Write-Host $detail
        if ($metricsLine) { Write-Host $metricsLine -ForegroundColor Green }
    } catch { Write-Host "Waiting for status: $($_.Exception.Message)" }
    if ($Once) { break }
    Start-Sleep -Seconds 3
} while ($true)

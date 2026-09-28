param([switch]$Once)
$projectRoot = Split-Path -Parent $PSScriptRoot
function Read-PairedJson([string]$Path) {
    $stream = [System.IO.File]::Open($Path,'Open','Read',([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    try { $reader = [System.IO.StreamReader]::new($stream); return ($reader.ReadToEnd() | ConvertFrom-Json) }
    finally { if ($reader) { $reader.Dispose() }; $stream.Dispose() }
}
do {
    try {
        $registration = Read-PairedJson (Join-Path $projectRoot 'outputs/orchestration/class_assisted_height_active.json')
        $state = Read-PairedJson (Join-Path $registration.run_dir 'status.json')
        $fraction = if ($state.stage -eq 'complete') { 1.0 } elseif ($state.total) { [double]$state.completed/[double]$state.total } else { 0.0 }
        $fraction = [Math]::Min(1.0,[Math]::Max(0.0,$fraction))
        $filled = [int]($fraction*30)
        $bar = '['+('#'*$filled)+('-'*(30-$filled))+']'
        $label = '{0} {1} ep {2} {3} {4:N1}%' -f $state.stage,$state.arm,$state.epoch,$bar,($fraction*100)
        if (-not $Once) { Clear-Host }
        Write-Host $label -ForegroundColor Cyan
        if ($state.error) { Write-Host $state.error -ForegroundColor Red }
        else { Write-Host 'A: uniform classes | B: predicted classes | Production unchanged' }
    } catch { Write-Host "Waiting for saved status: $($_.Exception.Message)" }
    if ($Once) { break }
    Start-Sleep -Seconds 3
} while ($true)

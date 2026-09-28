param([string]$Resume = '')
$ErrorActionPreference = 'Stop'
$dwProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $dwProjectRoot
$dwRunning = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like 'python*' -and $_.CommandLine -match 'scripts[\\/](train_[^\s]+\.py|class_assisted_height_proof\.py|audit_height_crop_context\.py)'
}
if ($dwRunning) { throw 'A training/GPU audit job is already active; no competing job will start.' }
$dwPython = Join-Path $dwProjectRoot '.venv\Scripts\python.exe'
if ($Resume) { & $dwPython scripts/train_height_band_sampling.py --resume $Resume }
else { & $dwPython scripts/train_height_band_sampling.py }
if ($LASTEXITCODE -ne 0) { throw 'Sampler experiment failed. Inspect saved failure before compatible resume.' }

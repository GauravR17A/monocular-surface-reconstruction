param([string]$Resume = '')
$ErrorActionPreference = 'Stop'
$dwProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $dwProjectRoot
$dwTrainers = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like 'python*' -and $_.CommandLine -match 'scripts[\\/](train_[^\s]+\.py|class_assisted_height_proof\.py)'
}
if ($dwTrainers) { throw 'A trainer is already active. No competing GPU job will be started.' }
$dwPython = Join-Path $dwProjectRoot '.venv\Scripts\python.exe'
if ($Resume) { & $dwPython scripts/train_class_assisted_height_fast.py --resume $Resume }
else { & $dwPython scripts/train_class_assisted_height_fast.py }
if ($LASTEXITCODE -ne 0) { throw 'Fast comparison stopped. Inspect saved error and bindings before resuming.' }

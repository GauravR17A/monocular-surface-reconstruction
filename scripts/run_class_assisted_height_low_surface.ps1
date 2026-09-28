param([string]$Resume = '')
$ErrorActionPreference = 'Stop'
$dwProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $dwProjectRoot
$dwTrainers = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like 'python*' -and $_.CommandLine -match 'scripts[\\/](train_[^\s]+\.py|class_assisted_height_proof\.py|analyze_class_assisted_low_surfaces\.py)'
}
if ($dwTrainers) { throw 'A GPU training/audit job is active. No competing job will be started.' }
$dwPython = Join-Path $dwProjectRoot '.venv\Scripts\python.exe'
if ($Resume) { & $dwPython scripts/train_class_assisted_height_low_surface.py --resume $Resume }
else { & $dwPython scripts/train_class_assisted_height_low_surface.py }
if ($LASTEXITCODE -ne 0) { throw 'Low-surface experiment stopped. Inspect saved failure and bindings before resume.' }

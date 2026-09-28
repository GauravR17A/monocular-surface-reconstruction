param([string]$Resume = '')
$ErrorActionPreference = 'Stop'
$dwProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $dwProjectRoot
$dwRunning = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like 'python*' -and $_.CommandLine -match 'scripts[\\/](train_[^\s]+\.py|class_assisted_height_proof\.py|prove_semantic_expert_height\.py|audit_height_[^\s]+\.py)'
}
if ($dwRunning) { throw 'A training/GPU audit job is active. No competing job will start.' }
$dwPython = Join-Path $dwProjectRoot '.venv\Scripts\python.exe'
if ($Resume) { & $dwPython scripts/train_semantic_expert_height.py --resume $Resume }
else { & $dwPython scripts/train_semantic_expert_height.py }
if ($LASTEXITCODE -ne 0) { throw 'Semantic expert experiment failed; inspect evidence before a compatible resume.' }

param([string]$Resume = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$activeTrainers = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like 'python*' -and ($_.CommandLine -match 'scripts[\\/](train_[^\s]+\.py|class_assisted_height_proof\.py)')
}
if ($activeTrainers) { throw 'A trainer is already active. Do not start a competing GPU job.' }
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if ($Resume) { & $pythonPath scripts/train_class_assisted_height.py --resume $Resume }
else {
    if (Test-Path -LiteralPath 'outputs/orchestration/class_assisted_height_active.json') {
        throw 'An existing development run is registered. Inspect its status and use -Resume with the exact directory.'
    }
    & $pythonPath scripts/train_class_assisted_height.py
}
if ($LASTEXITCODE -ne 0) { throw 'Paired run stopped. Inspect the saved error before any compatible resume.' }

param([string]$Resume = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$trainingPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$preparing = Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*prepare_rgb_multiregion_v3.py*' }
if ($preparing) { throw 'Preparation is already active. Wait for it to finish; do not start another downloader.' }
$activeTraining = Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*train_rgb_multiregion_v3.py*' }
if ($activeTraining) { throw 'This training run is already active.' }
if (-not (Test-Path -LiteralPath 'D:\MSRData\training\oem_multiregion_v3\manifest.json')) {
    & $trainingPython scripts/prepare_rgb_multiregion_v3.py
    if ($LASTEXITCODE -ne 0) { throw 'Preparation failed; downloaded files remain safely resumable.' }
}
if ($Resume) {
    & $trainingPython scripts/train_rgb_multiregion_v3.py --resume $Resume
} else {
    if (Test-Path -LiteralPath 'outputs\orchestration\rgb_multiregion_v3_active.json') {
        throw 'An existing run is registered. Inspect it and use -Resume with its exact experiment directory.'
    }
    & $trainingPython scripts/train_rgb_multiregion_v3.py
}
if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect logs and resume only the compatible committed checkpoint.' }

param([ValidateSet("cpu","cuda")][string]$Device = "cuda")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$pythonPath = Join-Path $projectRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Create .venv and install the dependencies in README.md first." }
& $pythonPath scripts/serve_api.py --device $Device

param(
    [Parameter(Mandatory = $true)]
    [string]$CheckpointPath
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $projectRoot "outputs\runtime"

if (-not [System.IO.Path]::IsPathRooted($CheckpointPath)) {
    $CheckpointPath = Join-Path $projectRoot $CheckpointPath
}
$CheckpointPath = [System.IO.Path]::GetFullPath($CheckpointPath)

if (-not (Test-Path -LiteralPath $CheckpointPath -PathType Leaf)) {
    throw "Checkpoint does not exist: $CheckpointPath"
}
if ([System.IO.Path]::GetExtension($CheckpointPath) -ne ".pt") {
    throw "Showcase checkpoint must be a .pt file: $CheckpointPath"
}

New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
$pointerPath = Join-Path $runtimeRoot "showcase_checkpoint.txt"
$CheckpointPath | Out-File -LiteralPath $pointerPath -Encoding utf8

Write-Host "Showcase checkpoint selected:" -ForegroundColor Green
Write-Host $CheckpointPath
Write-Host "Restart the prototype to load it."

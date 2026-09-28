param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [string]$CheckpointPath = ""
)

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Set-Location $ProjectRoot

Write-Host "Monocular Surface Reconstruction live training terminal" -ForegroundColor Cyan
if ($CheckpointPath) {
    Write-Host "Resuming checkpoint: $CheckpointPath"
} else {
    Write-Host "Starting full-data training: $ConfigPath"
}
Write-Host ""
$trainingArguments = @("scripts\train.py", "--config", $ConfigPath)
if ($CheckpointPath) {
    $trainingArguments += @("--resume", $CheckpointPath)
}
& $Python @trainingArguments
$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "Training finished successfully." -ForegroundColor Green
} else {
    Write-Host "Training stopped with exit code $exitCode." -ForegroundColor Red
}
Write-Host "You may close this window after reading the result."

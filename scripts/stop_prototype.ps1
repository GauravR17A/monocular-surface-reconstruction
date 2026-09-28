$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $projectRoot "outputs\runtime"
$processSnapshot = @(Get-CimInstance Win32_Process)

function Stop-RecordedProcessTree {
    param(
        [string]$PidFile,
        [string[]]$RequiredMarkers
    )

    if (-not (Test-Path -LiteralPath $PidFile)) {
        return
    }

    $recordedPid = 0
    if (-not [int]::TryParse((Get-Content -LiteralPath $PidFile -Raw).Trim(), [ref]$recordedPid)) {
        Write-Warning "Ignoring invalid PID file: $PidFile"
        Remove-Item -LiteralPath $PidFile -Force
        return
    }

    $rootProcess = $processSnapshot | Where-Object { $_.ProcessId -eq $recordedPid } | Select-Object -First 1
    if ($null -eq $rootProcess) {
        Remove-Item -LiteralPath $PidFile -Force
        return
    }

    $commandLine = [string]$rootProcess.CommandLine
    $matches = $false
    foreach ($marker in $RequiredMarkers) {
        if ($commandLine -like "*$marker*") {
            $matches = $true
            break
        }
    }
    if (-not $matches) {
        Write-Warning "PID $recordedPid no longer belongs to Monocular Surface Reconstruction; it was not stopped."
        Remove-Item -LiteralPath $PidFile -Force
        return
    }

    $tree = [System.Collections.Generic.List[int]]::new()
    function Add-Descendants {
        param([int]$ParentPid)
        foreach ($child in $processSnapshot | Where-Object { $_.ParentProcessId -eq $ParentPid }) {
            Add-Descendants -ParentPid ([int]$child.ProcessId)
            $tree.Add([int]$child.ProcessId)
        }
    }

    Add-Descendants -ParentPid $recordedPid
    $tree.Add($recordedPid)
    foreach ($processId in $tree) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }

    Remove-Item -LiteralPath $PidFile -Force
    Write-Host "Stopped recorded Monocular Surface Reconstruction process tree rooted at PID $recordedPid."
}

Stop-RecordedProcessTree `
    -PidFile (Join-Path $runtimeRoot "viewer.pid") `
    -RequiredMarkers @("npm.cmd", "npm-cli.js", "run dev", "run start")
Stop-RecordedProcessTree `
    -PidFile (Join-Path $runtimeRoot "api.pid") `
    -RequiredMarkers @("scripts\serve_api.py", "scripts/serve_api.py")

Write-Host "Monocular Surface Reconstruction prototype stop request completed." -ForegroundColor Green

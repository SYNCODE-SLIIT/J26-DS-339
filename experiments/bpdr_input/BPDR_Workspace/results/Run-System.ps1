param([int]$Port = 8501)
$ErrorActionPreference='Stop'
& (Join-Path (Split-Path -Parent $PSScriptRoot) 'Run-System.ps1') -Port $Port
exit $LASTEXITCODE

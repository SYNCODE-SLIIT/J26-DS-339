param([string]$DatasetRoot, [switch]$Execute, [switch]$Check, [switch]$NoBrowser, [int]$Port = 8888)
$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot 'BPDR_Workspace/Run-Notebooks.ps1') -DatasetRoot $DatasetRoot -Execute:$Execute -Check:$Check -NoBrowser:$NoBrowser -Port $Port
exit $LASTEXITCODE

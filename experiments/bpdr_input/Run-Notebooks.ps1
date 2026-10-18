param([string]$DatasetRoot)
$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot 'BPDR_Workspace/Run-Notebooks.ps1') -DatasetRoot $DatasetRoot
exit $LASTEXITCODE

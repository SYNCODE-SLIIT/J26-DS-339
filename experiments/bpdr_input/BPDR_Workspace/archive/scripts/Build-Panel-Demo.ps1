param([switch]$VerifyOnly, [switch]$VerifySources)
$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$componentRoot = [IO.Path]::GetFullPath((Join-Path $workspaceRoot '..'))
$env:PYTHONPATH = Join-Path $workspaceRoot 'source'
$taskArguments = @('-m', 'bpdr_input.reporting.panel_demo')
if ($VerifyOnly) { $taskArguments += '--verify-only' }
if ($VerifySources) { $taskArguments += '--verify-sources' }
& (Join-Path $componentRoot '.venv/Scripts/python.exe') @taskArguments
exit $LASTEXITCODE

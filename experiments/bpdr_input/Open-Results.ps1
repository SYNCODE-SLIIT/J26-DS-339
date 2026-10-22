$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot 'BPDR_Workspace/Open-Results.ps1')
exit $LASTEXITCODE

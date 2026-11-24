param(
    [int]$Port = 8515
)

$scriptPath = Join-Path $PSScriptRoot "BPDR_Workspace\Run-Component1-Panel.ps1"
& $scriptPath -Port $Port
exit $LASTEXITCODE

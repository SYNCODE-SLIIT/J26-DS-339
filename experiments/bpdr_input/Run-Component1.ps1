param(
    [ValidateRange(1, 65535)][int]$Port = 8505,
    [switch]$NoBrowser
)

$scriptPath = Join-Path $PSScriptRoot "BPDR_Workspace\Run-Component1.ps1"
& $scriptPath -Port $Port -NoBrowser:$NoBrowser
exit $LASTEXITCODE

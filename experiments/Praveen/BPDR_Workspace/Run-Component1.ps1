param(
    [ValidateRange(1, 65535)][int]$Port = 8505,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$componentRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$venvPython = Join-Path $componentRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "The component Python environment is missing at: $venvPython"
}

$env:PYTHONPATH = Join-Path $PSScriptRoot 'source'
$env:PYTHONDONTWRITEBYTECODE = '1'
$url = "http://127.0.0.1:$Port"

Push-Location -LiteralPath $PSScriptRoot
try {
    Write-Host "Component 1 (BPDR + Active Intent Confirmation): $url"
    Write-Host "Panel Research View: $url/research"
    if (-not $NoBrowser) {
        Start-Job -ScriptBlock {
            param($target)
            Start-Sleep -Seconds 2
            Start-Process $target
        } -ArgumentList $url | Out-Null
    }
    & $venvPython -X utf8 -B -m bpdr_input.web.app --host 127.0.0.1 --port $Port
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode

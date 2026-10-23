param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765
)

$ErrorActionPreference = 'Stop'
$repoDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Demo Python environment missing: pp1_demo\.venv\Scripts\python.exe. See pp1_demo\README.md.'
}

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
if (-not $env:PP1_BPDR_PYTHON) {
    $env:PP1_BPDR_PYTHON = $python
}

Push-Location -LiteralPath $repoDir
try {
    & $python -B -m pp1_demo.app --port $Port
    if ($LASTEXITCODE -ne 0) {
        throw "Harmony Lab exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

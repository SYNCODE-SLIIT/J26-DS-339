param([ValidateRange(1, 65535)][int]$Port = 8503)
$ErrorActionPreference = 'Stop'
$reviewComponentRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$reviewPython = Join-Path $reviewComponentRoot '.venv/Scripts/python.exe'
$reviewEntry = Join-Path $PSScriptRoot 'source/bpdr_input/application/review_demo_app.py'
if (-not (Test-Path -LiteralPath $reviewPython)) { throw 'The component Python environment is missing.' }
$env:PYTHONPATH = Join-Path $PSScriptRoot 'source'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
Push-Location -LiteralPath $PSScriptRoot
try {
    Write-Host "BPDR review demo: http://127.0.0.1:$Port"
    & $reviewPython -X utf8 -B -m streamlit run $reviewEntry --server.address 127.0.0.1 --server.port $Port --server.headless true
    $reviewExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $reviewExitCode

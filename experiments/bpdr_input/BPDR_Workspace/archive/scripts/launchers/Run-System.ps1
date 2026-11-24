param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
$componentRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

$env:PYTHONPATH = Join-Path $componentRoot 'BPDR_Workspace/source'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
Push-Location -LiteralPath $componentRoot
try {
    & (Join-Path $componentRoot '.venv/Scripts/python.exe') -m streamlit run (Join-Path $componentRoot 'BPDR_Workspace/source/bpdr_input/application/app.py') --server.address 127.0.0.1 --server.port $Port --server.headless true
    $systemExit = $LASTEXITCODE
} finally { Pop-Location }
exit $systemExit

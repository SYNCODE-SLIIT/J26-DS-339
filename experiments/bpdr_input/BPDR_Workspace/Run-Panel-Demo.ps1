param([switch]$Build, [switch]$Check, [switch]$Rehearse, [switch]$Offline, [int]$Port = 8501)
$ErrorActionPreference = 'Stop'
$componentRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Join-Path $componentRoot '.venv/Scripts/python.exe'
$env:PYTHONPATH = Join-Path $PSScriptRoot 'source'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
if (-not (Test-Path -LiteralPath $python)) { throw 'The component Python environment is missing.' }
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($Build) {
        & $python -B -m bpdr_input.reporting.panel_demo
        if ($LASTEXITCODE -ne 0) { throw 'Panel evidence build failed. Inspect the reported reason.' }
    }
    if ($Rehearse) {
        & $python -B -m bpdr_input.reporting.panel_rehearsal
        if ($LASTEXITCODE -ne 0) { throw 'Rehearsal did not meet its expected outcomes. Reports are retained under results/panel_demo/rehearsal.' }
    }
    if ($Check) {
        & $python -B -c 'from bpdr_input.reporting.panel_demo import load_verified_manifest; m=load_verified_manifest(verify_sources=True); print("Verified panel cases:", len(m["cases"]))'
        if ($LASTEXITCODE -ne 0) { throw 'Panel evidence verification failed.' }
        return
    }
    if ($Offline) {
        Invoke-Item -LiteralPath (Join-Path $PSScriptRoot 'results/panel_demo/index.html')
        return
    }
    Write-Host "Panel demo: http://127.0.0.1:$Port"
    & $python -B -m streamlit run (Join-Path $PSScriptRoot 'source/bpdr_input/application/demo_app.py') --server.address 127.0.0.1 --server.port $Port --server.headless true --theme.base light
    if ($LASTEXITCODE -ne 0) { throw 'The panel interface stopped with an error.' }
} finally { Pop-Location }

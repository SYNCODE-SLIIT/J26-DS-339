param(
    [int]$Port = 8515
)
$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Workspace
$env:PYTHONPATH = "$Workspace\source;$Workspace"

$VenvPython = Join-Path $Workspace "..\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "The component Python environment (with torch) is missing at: $VenvPython"
}
$PythonExe = $VenvPython

Write-Host "Starting J26-DS-339 Component 1 Research Panel Demonstration App (FastTNet Primary Model) on port $Port..." -ForegroundColor Cyan
& $PythonExe -X utf8 -B -m streamlit run "$Workspace\source\bpdr_input\application\component1_panel_app.py" --server.port $Port

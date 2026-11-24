param([switch]$Verify)
$ErrorActionPreference = 'Stop'
$notebookPython = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
$notebookIndex = Join-Path $PSScriptRoot 'notebooks\index.html'
if ($Verify) {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'source'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    & $notebookPython -X utf8 -B (Join-Path $PSScriptRoot 'tools\execute_research_notebooks.py')
    if ($LASTEXITCODE -ne 0) { throw 'Notebook verification failed. Check the retained failure output.' }
}
if (-not (Test-Path -LiteralPath $notebookIndex)) { throw 'Reading views are missing. Run Open-Notebooks.ps1 -Verify with the local assets available.' }
Start-Process -FilePath $notebookIndex

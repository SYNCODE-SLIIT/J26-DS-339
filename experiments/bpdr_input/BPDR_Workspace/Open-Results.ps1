$ErrorActionPreference = 'Stop'
$componentRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

$resultsPath = Join-Path $componentRoot 'BPDR_Workspace/results/index.html'
if (-not (Test-Path -LiteralPath $resultsPath -PathType Leaf)) { throw 'Results index is missing.' }
Start-Process -FilePath $resultsPath

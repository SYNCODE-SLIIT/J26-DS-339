$ErrorActionPreference = 'Stop'
$componentRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$artifacts = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'CURRENT_ARTIFACTS.json') | ConvertFrom-Json
foreach ($notebook in $artifacts.notebooks) {
    $reportPath = Join-Path $componentRoot $notebook.executed_html
    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) { throw "Saved report is missing: $reportPath" }
    Start-Process -FilePath $reportPath
}

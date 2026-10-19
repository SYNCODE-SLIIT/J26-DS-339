param([string]$DatasetRoot, [switch]$Execute, [switch]$Check, [switch]$NoBrowser, [int]$Port = 8888)
$ErrorActionPreference = 'Stop'
$componentRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Join-Path $componentRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'The local Python environment is missing. Restore the component .venv before opening these notebooks.'
}
if ($Execute -and $Check) { throw 'Choose either -Execute or -Check.' }
if (-not $DatasetRoot) { $DatasetRoot = Join-Path $PSScriptRoot 'datasets\vocadito' }
$env:PYTHONPATH = Join-Path $PSScriptRoot 'source'
$env:PYTHONDONTWRITEBYTECODE = '1'
$command = if ($Execute) { 'execute' } elseif ($Check) { 'check' } else { 'launch' }
$arguments = @('-P', '-B', '-m', 'bpdr_input.reporting.notebooks', $command, '--dataset-root', $DatasetRoot, '--port', $Port)
if ($NoBrowser) { $arguments += '--no-browser' }
& $python @arguments
exit $LASTEXITCODE

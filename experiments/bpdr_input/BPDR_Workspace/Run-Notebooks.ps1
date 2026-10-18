param([string]$DatasetRoot)
$ErrorActionPreference = 'Stop'
$componentRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

if (-not $DatasetRoot) { $DatasetRoot = Join-Path $componentRoot 'data\vocadito' }
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$uvExecutable = if ($uvCommand) { $uvCommand.Source } else { Join-Path $componentRoot '.tools\uv.exe' }
if (-not (Test-Path -LiteralPath $uvExecutable)) { throw 'Install uv from https://docs.astral.sh/uv/getting-started/installation/' }
& $uvExecutable run --project $componentRoot --locked --extra rendering --group notebooks --group presentation python -m bpdr_input.reporting.notebooks launch --dataset-root $DatasetRoot
exit $LASTEXITCODE

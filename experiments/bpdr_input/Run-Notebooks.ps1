param([string]$DatasetRoot)
$ErrorActionPreference = 'Stop'
if (-not $DatasetRoot) { $DatasetRoot = Join-Path $PSScriptRoot 'data\vocadito' }
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$uvExecutable = if ($uvCommand) { $uvCommand.Source } else { Join-Path $PSScriptRoot '.tools\uv.exe' }
if (-not (Test-Path -LiteralPath $uvExecutable)) { throw 'Install uv from https://docs.astral.sh/uv/getting-started/installation/' }
& $uvExecutable run --project $PSScriptRoot --locked --group notebooks python -m bpdr_input.notebooks launch --dataset-root $DatasetRoot
exit $LASTEXITCODE

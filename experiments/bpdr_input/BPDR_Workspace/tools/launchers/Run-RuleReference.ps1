param(
    [Parameter(Mandatory=$true)][string]$Source,
    [switch]$HandOffPrepared
)
# Source: one WAV/FLAC recording, or a demo input folder written by `python -m bpdr_input.demo build-set`.
$ErrorActionPreference = 'Stop'
$componentRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$uvExecutable = if ($uvCommand) { $uvCommand.Source } else { Join-Path $componentRoot '.tools\uv.exe' }
if (-not (Test-Path -LiteralPath $uvExecutable)) {
    throw 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/ and run this script again.'
}
$arguments = @('run', '--project', $componentRoot, '--extra', 'rendering', 'python', '-m', 'bpdr_input.demo', 'run', $Source)
if ($HandOffPrepared) { $arguments += '--hand-off-prepared' }
& $uvExecutable @arguments
exit $LASTEXITCODE

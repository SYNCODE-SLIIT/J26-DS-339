param([Parameter(Mandatory=$true)][string]$InputAudio)
$ErrorActionPreference = 'Stop'
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$uvExecutable = if ($uvCommand) { $uvCommand.Source } else { Join-Path $PSScriptRoot '.tools\uv.exe' }
if (-not (Test-Path -LiteralPath $uvExecutable)) {
    throw 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/ and run this script again.'
}
& $uvExecutable run --project $PSScriptRoot bpdr-process $InputAudio
exit $LASTEXITCODE

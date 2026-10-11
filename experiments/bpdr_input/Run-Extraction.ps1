param([int]$Limit, [string]$First, [int]$Threads, [int]$SeparatorOverlap,
      [switch]$RetryFailed, [switch]$IndexOnly, [switch]$VocalOnly, [switch]$KeepStems)
$ErrorActionPreference = 'Stop'
# Runs the mirrored melody-extraction pipeline through extraction\run_local.py. -IndexOnly needs no models.
$python = Join-Path $PSScriptRoot 'checkpoints\melody-extraction\.venv-run\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    if (-not $IndexOnly) { throw 'Extraction tools are not installed. Run .\Setup-ExtractionTools.ps1 first, or use -IndexOnly.' }
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
$arguments = @('-X', 'utf8', (Join-Path $PSScriptRoot 'extraction\run_local.py'))
if ($IndexOnly) { $arguments += '--index-only' }
if ($VocalOnly) { $arguments += '--vocal-only' }
if ($KeepStems) { $arguments += '--keep-stems' }
if ($First) { $arguments += @('--first', $First) }
if ($Threads) { $arguments += @('--threads', $Threads) }
if ($SeparatorOverlap) { $arguments += @('--separator-overlap', $SeparatorOverlap) }
if ($Limit) { $arguments += @('--limit', $Limit) }
if ($RetryFailed) { $arguments += '--retry-failed' }
& $python @arguments
exit $LASTEXITCODE

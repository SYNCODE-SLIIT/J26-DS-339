param([int]$Limit, [string]$First, [int]$Threads, [int]$SeparatorOverlap,
      [switch]$Gpu, [switch]$RetryFailed, [switch]$IndexOnly, [switch]$VocalOnly, [switch]$KeepStems)
$ErrorActionPreference = 'Stop'
$componentRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))

# Runs the mirrored melody-extraction pipeline through extraction\run_local.py. -IndexOnly needs no models.
# -Gpu uses the Intel-GPU environment (.venv-xpu) for the vocal separator.
$environment = if ($Gpu) { '.venv-xpu' } else { '.venv-run' }
$python = Join-Path $componentRoot "checkpoints\melody-extraction\$environment\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    if (-not $IndexOnly) { throw "The $environment environment is not installed. Run .\Setup-ExtractionTools.ps1 first, or use -IndexOnly." }
    $python = Join-Path $componentRoot '.venv\Scripts\python.exe'
}
$arguments = @('-X', 'utf8', (Join-Path $componentRoot 'extraction\run_local.py'))
if ($IndexOnly) { $arguments += '--index-only' }
if ($VocalOnly) { $arguments += '--vocal-only' }
if ($KeepStems) { $arguments += '--keep-stems' }
if ($Gpu) { $arguments += @('--device', 'xpu') }
if ($First) { $arguments += @('--first', $First) }
if ($Threads) { $arguments += @('--threads', $Threads) }
if ($SeparatorOverlap) { $arguments += @('--separator-overlap', $SeparatorOverlap) }
if ($Limit) { $arguments += @('--limit', $Limit) }
if ($RetryFailed) { $arguments += '--retry-failed' }
& $python @arguments
exit $LASTEXITCODE

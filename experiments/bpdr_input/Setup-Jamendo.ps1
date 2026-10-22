param([string]$SourceDataset)
$ErrorActionPreference = 'Stop'
# Mirror the shared MTG-Jamendo metadata and melody-extraction-pipeline into this component's ignored data folder.
# The older melody-extraction pilot is deliberately not mirrored. Safe to run again: it refreshes code and metadata only.
if (-not $SourceDataset) { $SourceDataset = Join-Path $PSScriptRoot '..\..\datasets\mtg-jamendo' }
$source = (Resolve-Path -LiteralPath $SourceDataset).Path
$target = Join-Path $PSScriptRoot 'data\mtg-jamendo'
New-Item -ItemType Directory -Force $target | Out-Null
foreach ($name in 'annotations', 'splits') {
    robocopy (Join-Path $source $name) (Join-Path $target $name) /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copy failed: $name" }
}
foreach ($name in 'LICENSE', 'README.md', 'audio_licenses.txt') {
    Copy-Item -LiteralPath (Join-Path $source $name) -Destination $target -Force
}
# Pipeline code only. outputs/, work/, manifest.csv, catalog.json and history.log must come from a real run.
robocopy (Join-Path $source 'melody-extraction-pipeline') (Join-Path $target 'melody-extraction-pipeline') /E `
    /XD outputs work __pycache__ /XF manifest.csv catalog.json history.log /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw 'Copy failed: melody-extraction-pipeline' }
$audio = Join-Path $target 'vocal-tagged-audio'
$earlier = Join-Path $PSScriptRoot 'data\vocal-tagged-audio'
if (-not (Test-Path -LiteralPath $audio)) {
    if (Test-Path -LiteralPath $earlier) { Move-Item -LiteralPath $earlier -Destination $audio }
    else { Write-Warning "No audio found. Put the MTG-Jamendo NN\*.low.mp3 folders in $audio" }
}
$songs = if (Test-Path -LiteralPath $audio) { @(Get-ChildItem -LiteralPath $audio -Recurse -File -Filter '*.low.mp3').Count } else { 0 }
Write-Output "MTG-Jamendo mirror: $target"
Write-Output "Local MP3s: $songs"
exit 0

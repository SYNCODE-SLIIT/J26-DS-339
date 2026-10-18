param([ValidateSet('Vocal', 'Full')][string]$Stage = 'Vocal', [switch]$IntelGpu)
$ErrorActionPreference = 'Stop'
$componentRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))

$ProgressPreference = 'SilentlyContinue'
# Installs the model tools the mirrored melody-extraction pipeline expects, in its own asset layout, under the
# ignored checkpoints\melody-extraction folder. Commits and SHA-256 values are the ones pinned in the team's
# provenance files; a mismatch stops the setup. Safe to run again: finished steps are skipped.
#   -Stage Vocal   FFmpeg, GAME and the BS-RoFormer separator: enough for Run-Extraction.ps1 -VocalOnly
#   -Stage Full    adds SheetSage2 and its MERT parent (about 2.6 GB) for the complete pipeline
#   -IntelGpu      adds a second launcher environment with the Intel-GPU build of PyTorch (about 1.4 GB)
#                  for Run-Extraction.ps1 -Gpu and for training with --device xpu
$assets = Join-Path $componentRoot 'checkpoints\melody-extraction'
$models = Join-Path $assets 'models'
$tools = Join-Path $assets 'tools'
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$uv = if ($uvCommand) { $uvCommand.Source } else { Join-Path $componentRoot '.tools\uv.exe' }
if (-not (Test-Path -LiteralPath $uv)) { throw 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/' }
New-Item -ItemType Directory -Force $models, $tools | Out-Null
$downloads = [ordered]@{}

function Assert-Native([string]$What) { if ($LASTEXITCODE) { throw "$What failed with exit code $LASTEXITCODE" } }

function Confirm-Hash([string]$Path, [string]$Sha256) {
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower()
    if ($Sha256 -and $actual -ne $Sha256) { throw "SHA-256 mismatch for $Path. Expected $Sha256, found $actual." }
    $downloads[(Split-Path -Leaf (Split-Path -Parent $Path)) + '/' + (Split-Path -Leaf $Path)] = $actual
}

function Get-Checked([string]$Url, [string]$Path, [string]$Sha256) {
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "Downloading $(Split-Path -Leaf $Path)"
        & curl.exe --location --fail --silent --show-error --output "$Path.part" $Url
        Assert-Native "Download of $Url"
        Move-Item -LiteralPath "$Path.part" -Destination $Path
    }
    Confirm-Hash $Path $Sha256
}

function New-Environment([string]$Name, [string]$Python) {
    $venv = Join-Path $assets $Name
    if (-not (Test-Path -LiteralPath (Join-Path $venv 'Scripts\python.exe'))) {
        & $uv venv --quiet --python $Python $venv
        Assert-Native "Creating $Name"
    }
    # The pipeline hard-codes the POSIX bin/python path; a junction makes it resolve on Windows.
    if (-not (Test-Path -LiteralPath (Join-Path $venv 'bin'))) {
        New-Item -ItemType Junction -Path (Join-Path $venv 'bin') -Target (Join-Path $venv 'Scripts') | Out-Null
    }
    return (Join-Path $venv 'Scripts\python.exe')
}

function Install-Packages([string]$Python, [string]$What, [string[]]$Arguments) {
    Write-Host "Installing $What"
    # Some networks reset connections to the package hosts; finished downloads are cached, so a retry continues.
    foreach ($attempt in 1..5) {
        & $uv pip install --quiet --python $Python @Arguments
        if ($LASTEXITCODE -eq 0) { return }
        Write-Host "Installing $What was interrupted (attempt $attempt); retrying"
    }
    throw "Installing $What failed after 5 attempts"
}

function Save-Environment([string]$Name, [string]$Python) {
    $frozen = & $uv pip freeze --python $Python
    Assert-Native "Recording $Name"
    $frozen | Set-Content -Encoding utf8 (Join-Path $assets "environment$Name.txt")
}

$cpuTorch = @('--index-url', 'https://download.pytorch.org/whl/cpu')

# FFmpeg: the pipeline calls ffmpeg and ffprobe. Version and checksum come from gyan.dev; the file itself comes
# from that build's GitHub mirror, which is much faster. The checksum only guards against a damaged download.
$ffmpegBin = Join-Path $tools 'ffmpeg\bin'
if (-not (Test-Path -LiteralPath (Join-Path $ffmpegBin 'ffprobe.exe'))) {
    $published = & curl.exe --location --fail --silent --show-error 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip.sha256'
    Assert-Native 'FFmpeg checksum lookup'
    $release = "$(& curl.exe --location --fail --silent --show-error 'https://www.gyan.dev/ffmpeg/builds/release-version')".Trim()
    Assert-Native 'FFmpeg version lookup'
    $archive = Join-Path $tools 'ffmpeg-release-essentials.zip'
    Get-Checked "https://github.com/GyanD/codexffmpeg/releases/download/$release/ffmpeg-$release-essentials_build.zip" $archive ("$published".Trim().ToLower())
    $unpacked = Join-Path $tools 'ffmpeg-unpacked'
    Expand-Archive -LiteralPath $archive -DestinationPath $unpacked -Force
    $probe = Get-ChildItem -LiteralPath $unpacked -Recurse -Filter 'ffprobe.exe' | Select-Object -First 1
    if (-not $probe) { throw 'The FFmpeg archive has no ffprobe.exe' }
    New-Item -ItemType Directory -Force $ffmpegBin | Out-Null
    foreach ($name in 'ffmpeg.exe', 'ffprobe.exe') { Copy-Item -LiteralPath (Join-Path $probe.DirectoryName $name) -Destination $ffmpegBin }
    Remove-Item -Recurse -Force $unpacked
}

# GAME: singing-note transcription, at the commit and release checkpoint the team pinned.
$game = Join-Path $tools 'GAME'
if (-not (Test-Path -LiteralPath (Join-Path $game 'infer.py'))) {
    & git clone --quiet https://github.com/openvpi/GAME $game
    Assert-Native 'Cloning GAME'
}
& git -C $game checkout --quiet f4239345f0fed21890d362c74a35fd857724aa82
Assert-Native 'Checking out the pinned GAME commit'
$gameModel = Join-Path $models 'GAME-1.0-medium'
$archive = Join-Path $models 'GAME-1.0-medium.zip'
# The archive value is the release asset's own digest on GitHub. The team's provenance labels its value as the
# archive hash, but it is the SHA-256 of model.pt inside the archive, so that is what is checked against it.
Get-Checked 'https://github.com/openvpi/GAME/releases/download/v1.0.0/GAME-1.0-medium.zip' $archive '8c5b3e531e2905b935e664e2f533921cd637243770fab5282413bdb5051ca60c'
if (-not (Test-Path -LiteralPath (Join-Path $gameModel 'model.pt'))) {
    Expand-Archive -LiteralPath $archive -DestinationPath $gameModel -Force
    $found = Get-ChildItem -LiteralPath $gameModel -Recurse -Filter 'model.pt' | Select-Object -First 1
    if (-not $found) { throw 'The GAME archive has no model.pt' }
    if ($found.DirectoryName -ne $gameModel) { Get-ChildItem -LiteralPath $found.DirectoryName | Move-Item -Destination $gameModel }
}
Confirm-Hash (Join-Path $gameModel 'model.pt') 'e9904159fb0646e1a352b9d2bc74615547cfa3e32d45c7464d440ac142846d93'
$gamePython = New-Environment '.venv-game' '3.12'
Install-Packages $gamePython 'PyTorch (CPU) for GAME' (@('torch==2.14.0') + $cpuTorch)
Install-Packages $gamePython 'GAME requirements' @('-r', (Join-Path $game 'requirements.txt'), 'numpy==1.26.4')
Save-Environment '.venv-game' $gamePython

# Vocal separator: same BS-RoFormer checkpoint as the Mac run, through the PyTorch implementation.
$separator = Join-Path $models 'mlx-audio-separator'
New-Item -ItemType Directory -Force $separator | Out-Null
Get-Checked 'https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/model_bs_roformer_ep_317_sdr_12.9755.ckpt' `
    (Join-Path $separator 'model_bs_roformer_ep_317_sdr_12.9755.ckpt') '5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa'
$runPython = New-Environment '.venv-run' '3.12'
Install-Packages $runPython 'PyTorch (CPU) for the launcher' (@('torch==2.14.0') + $cpuTorch)
# audio-separator 0.47.0 imports audioread without declaring it; librosa 1.0 no longer brings it in.
Install-Packages $runPython 'separator, MIDI and scoring packages' @('audio-separator[cpu]==0.47.0', 'audioread', 'pretty_midi', 'soundfile', 'mir_eval')
Save-Environment '.venv-run' $runPython

if ($IntelGpu) {
    # Same launcher packages on the Intel-GPU (xpu) build of PyTorch, kept apart from the verified CPU environment.
    $gpuPython = New-Environment '.venv-xpu' '3.12'
    Install-Packages $gpuPython 'PyTorch (Intel GPU) for the launcher' @('torch==2.14.1', '--index-url', 'https://download.pytorch.org/whl/xpu')
    Install-Packages $gpuPython 'separator, MIDI and scoring packages' @('audio-separator[cpu]==0.47.0', 'audioread', 'pretty_midi', 'soundfile', 'mir_eval')
    Save-Environment '.venv-xpu' $gpuPython
}

if ($Stage -eq 'Full') {
    # SheetSage2 and its MERT parent: pinned Hugging Face snapshots in the layout the worker reads offline.
    $sheetPython = New-Environment '.venv-sheetsage2' '3.11'
    Install-Packages $sheetPython 'Hugging Face client' @('huggingface-hub==0.36.0')
    $env:HF_HOME = Join-Path $models 'hf-home'
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
    # Some networks reset a share of connections to this host. snapshot_download then returns a partly filled
    # folder without an error, so every file in the pinned listing is fetched and confirmed one by one.
    $fetch = @'
import sys, time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
def retry(call, what):
    for attempt in range(1, 41):
        try:
            return call()
        except Exception as error:
            print(f'{what}: attempt {attempt} interrupted ({type(error).__name__}); retrying', file=sys.stderr, flush=True)
            time.sleep(4)
    raise SystemExit(f'Could not complete {what}')
def fetch(repository, revision, skip=()):
    names = retry(lambda: [item.rfilename for item in HfApi().model_info(repository, revision=revision).siblings], repository+' file list')
    names = [name for name in names if not name.startswith(skip)]
    paths = [Path(retry(lambda: hf_hub_download(repository, name, revision=revision), repository+'/'+name)) for name in names]
    root = paths[0].parents[len(Path(names[0]).parts)-1]
    missing = [name for name in names if not (root/name).is_file()]
    if missing:
        raise SystemExit(f'{repository} is incomplete: {missing}')
    return root
print(fetch('m-a-p/SheetSage2', '32c7c7473e7b59e1ed36bd8a8fefbd11ae8d468a', skip=('render_assets/', 'benchmarks/', 'tests/')))
print(fetch('m-a-p/MERT-v2-FullSong', 'd8ba1c745e733b3908ce6ad16ebeb17ac7600a42'))
'@
    Write-Host 'Downloading SheetSage2 and MERT-v2-FullSong snapshots'
    $snapshots = @(& $sheetPython -c $fetch)
    Assert-Native 'Downloading Hugging Face snapshots'
    Confirm-Hash (Join-Path $snapshots[-2] 'model.safetensors') 'b235f68091a5f5b644000f2b5acb57d1e70432aca2b34ab1b9cf27236e1f4274'
    Confirm-Hash (Join-Path $snapshots[-1] 'model.safetensors') 'e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d'
    Install-Packages $sheetPython 'PyTorch (CPU) for SheetSage2' (@('torch==2.8.0', 'torchaudio==2.8.0') + $cpuTorch)
    Install-Packages $sheetPython 'SheetSage2 requirements' @('-r', (Join-Path $snapshots[-2] 'requirements.txt'), 'soundfile')
    Save-Environment '.venv-sheetsage2' $sheetPython
}

$ffmpegVersion = (& (Join-Path $ffmpegBin 'ffmpeg.exe') -version | Select-Object -First 1)
[ordered]@{
    recorded_utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    stage = $Stage
    platform = [System.Environment]::OSVersion.VersionString
    device = if (Test-Path -LiteralPath (Join-Path $assets '.venv-xpu\Scripts\python.exe')) { 'CPU by default; Intel GPU (PyTorch xpu) through .venv-xpu when the launcher is run with --device xpu' } else { 'CPU' }
    ffmpeg = "$ffmpegVersion"
    sha256 = $downloads
    environments = 'environment.venv-*.txt beside this file'
    differences_from_team_provenance = @(
        'Vocal separator runs through audio-separator (PyTorch, CPU or Intel GPU) instead of mlx-audio-separator (MLX, Apple GPU); same checkpoint file.',
        'GAME and SheetSage2 run on CPU instead of the Apple GPU.',
        'FFmpeg is the Windows essentials build named above.',
        'Outputs are therefore not expected to be identical to the Mac run and must not be mixed with it without a comparison.'
    )
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $assets 'local-provenance.json')
Write-Output "Extraction tools ready ($Stage): $assets"
exit 0

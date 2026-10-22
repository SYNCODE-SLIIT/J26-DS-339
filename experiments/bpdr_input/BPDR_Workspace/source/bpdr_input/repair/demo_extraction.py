"""Bounded single-waveform connection to the existing pinned GAME extractor."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from bpdr_input import COMPONENT_ROOT
from bpdr_input.audio.contracts import AcceptedAudio, ProcessingResult, Status
from bpdr_input.audio.pipeline import checksum, validate_package


def run(handoff_directory, timeout_seconds=120):
    directory = Path(handoff_directory).resolve()
    workspace = (COMPONENT_ROOT / 'BPDR_Workspace').resolve()
    if not directory.is_relative_to(workspace):
        raise ValueError('Demo handoff must remain inside BPDR_Workspace.')
    data = json.loads((directory / 'handoff.json').read_text())
    accepted = data.get('accepted_audio')
    if not accepted or data['status'] not in ('ACCEPT_UNCHANGED', 'ACCEPT_PREPARED'):
        raise ValueError('Only an accepted package can reach the extractor.')
    audio = AcceptedAudio(accepted['path'], accepted['checksum'], accepted['sample_rate_hz'], accepted['sample_count'], accepted['channels'])
    package = ProcessingResult(data['request_id'], data['source_id'], Status(data['status']), audio)
    validate_package(package)
    if not Path(audio.path).resolve().is_relative_to(directory):
        raise ValueError('Selected audio leaves its reviewed package.')
    assets = COMPONENT_ROOT / 'checkpoints/melody-extraction'
    python = assets / '.venv-game/Scripts/python.exe'
    model = assets / 'models/GAME-1.0-medium/model.pt'
    runner = assets / 'tools/GAME'
    if not python.exists() or not model.exists():
        raise FileNotFoundError('The pinned local GAME runtime or model is unavailable.')
    attempt = directory / 'extraction_attempts' / uuid.uuid4().hex
    inputs = attempt / 'extractor_input'
    inputs.mkdir(parents=True)
    target = inputs / 'accepted.wav'
    shutil.copyfile(audio.path, target)
    if checksum(target) != audio.checksum:
        raise ValueError('Extractor input differs from the reviewed accepted waveform.')
    output = attempt / 'game_notes'
    seed = "import sys,random,numpy as np,torch,runpy;torch.set_num_threads(2);torch.manual_seed(339);np.random.seed(339);random.seed(339);sys.argv=sys.argv[1:];runpy.run_path('infer.py',run_name='__main__')"
    command = [str(python), '-B', '-c', seed, 'infer.py', 'extract', str(inputs), '-m', str(model), '--glob', '*.wav',
               '--output-formats', 'mid,csv', '--pitch-format', 'number', '--output-dir', str(output)]
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    started = time.perf_counter()
    receipt = dict(scope='One accepted demo waveform consumed by pinned GAME, not a musical-accuracy benchmark',
                   accepted_sha256=audio.checksum, model_sha256=checksum(model), seed=339, success=False,
                   attempt_directory=str(attempt))
    try:
        process = subprocess.run(command, cwd=runner, env=environment, timeout=timeout_seconds, capture_output=True, text=True, encoding='utf-8', errors='replace')
        (attempt / 'extractor_stdout.txt').write_text(process.stdout, encoding='utf-8')
        (attempt / 'extractor_stderr.txt').write_text(process.stderr, encoding='utf-8')
        files = list(output.rglob('*.csv')) + list(output.rglob('*.mid')) if output.exists() else []
        receipt.update(returncode=process.returncode, success=process.returncode == 0 and bool(files),
                       artifacts=[dict(path=str(path), sha256=checksum(path)) for path in files])
    except subprocess.TimeoutExpired:
        receipt.update(reason='EXTRACTOR_TIMEOUT', timeout_seconds=timeout_seconds)
    receipt['elapsed_seconds'] = round(time.perf_counter() - started, 3)
    (attempt / 'extractor_result.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (directory / 'extractor_result.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt
